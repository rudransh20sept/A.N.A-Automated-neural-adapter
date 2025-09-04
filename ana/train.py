# A.N.A - Automated Neural Adapter
# A GUI application for fine-tuning Language Models using LoRA.
# This version includes disk offloading for training very large models.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import torch
import sys
import traceback
import time
from datetime import datetime

# Core ML functionality imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset


def train_and_merge(
    model_id: str,
    merged_output_path: str,
    epochs: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: str,
    dataset_source: str,
    max_length: int,
    batch_size: int,
    learning_rate: float,
    use_paged_optimizer: bool,
    hf_token: str = None,
    offload_folder: str = None,  # NEW: Parameter for disk offload folder
    dataset_path: str = None,
    hf_dataset_id: str = None,
    hf_dataset_split: str = None,
    logger_callback=print,
):
    """
    The core function to load data, configure LoRA, train the model,
    and merge the adapter weights back into the base model.
    """
    auth_args = {"token": hf_token if hf_token else None}
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger_callback(f"🖥️ Using device: {device}")
    if device == "cuda":
        logger_callback(f" GPU: {torch.cuda.get_device_name(0)}")
        logger_callback(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # NEW: Create a dictionary for model loading arguments to keep code clean
    model_load_kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        **auth_args
    }
    
    # Use GPU if available, otherwise use CPU with disk offloading
    if device == "cuda":
        model_load_kwargs["device_map"] = "auto"
    else:
        model_load_kwargs["device_map"] = "auto"
        logger_callback("⚠️ CUDA not available, using CPU with disk offloading")
    
    # Add offload_folder only if it's provided
    if offload_folder:
        model_load_kwargs["offload_folder"] = offload_folder
        logger_callback(f" Disk offloading enabled. Using folder: {offload_folder}")

    logger_callback(" Loading dataset...")
    if dataset_source == "local":
        if not dataset_path or not os.path.exists(dataset_path):
            raise FileNotFoundError(f" Local dataset file not found at: {dataset_path}")
        dataset = load_dataset("text", data_files={"train": dataset_path})["train"]
        if "text" not in dataset.column_names:
            raise ValueError(" The local dataset file must contain a 'text' column.")
    elif dataset_source == "hub":
        if not hf_dataset_id or not hf_dataset_split:
            raise ValueError(" Hugging Face dataset ID and split must be provided.")
        logger_callback(f"⬇ Downloading split '{hf_dataset_split}' from '{hf_dataset_id}'...")
        full_dataset = load_dataset(hf_dataset_id, split=hf_dataset_split, **auth_args)

        def concatenate_columns(example):
            return {"text": " ".join(str(v) for v in example.values() if v is not None)}

        dataset = full_dataset.map(concatenate_columns, remove_columns=full_dataset.column_names)
        logger_callback(" Successfully loaded and prepared dataset from the Hub.")
    else:
        raise ValueError("Invalid dataset source specified.")

    lora_adapter_output_dir = os.path.join(merged_output_path, "lora_adapters_temp")
    os.makedirs(lora_adapter_output_dir, exist_ok=True)

    logger_callback(f" Loading base model and tokenizer from: {model_id}...")
    # UPDATED: Use the kwargs dictionary to load the model
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **auth_args)

    if tokenizer.pad_token is None:
        logger_callback("Tokenizer missing pad_token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logger_callback(" Configuring LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules.split(",") if target_modules else ["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger_callback(" Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    logger_callback(f" Setting up training for {epochs} epochs...")
    
    if use_paged_optimizer and device == "cuda":
        try:
            import bitsandbytes
            optim_choice = "paged_adamw_8bit"
            logger_callback(" Using 8-bit Paged AdamW optimizer to save VRAM.")
        except ImportError:
            logger_callback(" bitsandbytes not found! Falling back to standard optimizer.")
            logger_callback("   Please run 'pip install bitsandbytes' for VRAM savings.")
            optim_choice = "adamw_torch"
    elif use_paged_optimizer and device != "cuda":
        logger_callback(" 8-bit optimizer only available on CUDA devices. Using standard optimizer.")
        optim_choice = "adamw_torch"
    else:
        optim_choice = "adamw_torch"
        logger_callback("Using standard AdamW optimizer.")
        
    # Enable GPU training if available
    fp16 = torch.cuda.is_available()
    if fp16:
        logger_callback(" Using FP16 precision for faster training on GPU.")
    
    training_args = TrainingArguments(
        output_dir=lora_adapter_output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        optim=optim_choice,
        logging_dir=f"{lora_adapter_output_dir}/logs",
        report_to="none",
        seed=42,
        fp16=fp16,
        dataloader_pin_memory=True if device == "cuda" else False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    resume_from_checkpoint = os.path.isdir(lora_adapter_output_dir) and any(d.startswith("checkpoint-") for d in os.listdir(lora_adapter_output_dir))
    if resume_from_checkpoint:
        logger_callback(" Resuming training from checkpoint...")
    
    # Clear GPU cache before training
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Add training progress monitoring
    logger_callback(f" Starting training with {len(tokenized_dataset)} samples")
    logger_callback(f" Batch size: {batch_size}, Total steps: {len(tokenized_dataset) // batch_size * epochs}")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    logger_callback(f" Saving final LoRA adapter to: {lora_adapter_output_dir}")
    model.save_pretrained(lora_adapter_output_dir)
    tokenizer.save_pretrained(lora_adapter_output_dir)

    logger_callback(" Merging LoRA adapters with base model...")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # UPDATED: Use kwargs dictionary again when reloading model for merge
    base_model = AutoModelForCausalLM.from_pretrained(model_id, **model_load_kwargs)
    base_tokenizer = AutoTokenizer.from_pretrained(model_id, **auth_args)
    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_output_dir)
    merged_model = lora_model.merge_and_unload()

    logger_callback(f" Saving merged model to: {merged_output_path}")
    os.makedirs(merged_output_path, exist_ok=True)
    merged_model.save_pretrained(merged_output_path, safe_serialization=True)
    base_tokenizer.save_pretrained(merged_output_path)

    logger_callback("🎉 Done! Merged model is ready.")
    return merged_output_path


class ConsoleRedirector:
    def __init__(self, text_widget: tk.Text, original_stdout):
        self.text_widget, self.original_stdout = text_widget, original_stdout

    def write(self, message: str):
        self.original_stdout.write(message)
        self.text_widget.after(0, self._write_to_widget, message)

    def _write_to_widget(self, message: str):
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)

    def flush(self):
        self.original_stdout.flush()


class LoRATrainerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("A.N.A - Automated Neural Adapter -by Rudransh Joshi")
        self.root.geometry("1000x800")  # Wider window for two columns

        self._setup_theme()

        self.model_id = tk.StringVar(value="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.output_path = tk.StringVar(value=os.path.join(os.getcwd(), "merged_model_output"))
        self.hf_token = tk.StringVar(value="")
        self.epochs = tk.IntVar(value=3)
        self.lora_r = tk.IntVar(value=8)
        self.lora_alpha = tk.IntVar(value=16)
        self.lora_dropout = tk.DoubleVar(value=0.1)
        self.target_modules = tk.StringVar(value="q_proj,v_proj,k_proj,o_proj")
        self.max_length = tk.IntVar(value=256)
        self.batch_size = tk.IntVar(value=1)
        self.learning_rate = tk.DoubleVar(value=2e-4)
        self.dataset_source = tk.StringVar(value="local")
        self.local_dataset_path = tk.StringVar(value="")
        self.hf_dataset_id = tk.StringVar(value="HuggingFaceH4/ultrachat_200k")
        self.hf_dataset_split = tk.StringVar(value="train_sft")
        self.use_paged_optimizer = tk.BooleanVar(value=True)
        # NEW: Variables for disk offloading
        self.enable_disk_offload = tk.BooleanVar(value=False)
        self.offload_folder_path = tk.StringVar(value="")

        self.create_widgets()

    def _setup_theme(self):
        BG_PRIMARY, BG_SECONDARY, ACCENT, SELECT_BG = "#0d1b2a", "#1b263b", "#203650", "#485970"
        self.TEXT_COLOR = "#e0e1dd"
        self.root.configure(bg=BG_PRIMARY)
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', background=BG_PRIMARY, foreground=self.TEXT_COLOR, fieldbackground=BG_SECONDARY, borderwidth=1, relief=tk.FLAT)
        style.map('.', background=[('active', ACCENT)])
        style.configure('TFrame', background=BG_PRIMARY)
        style.configure('TLabel', background=BG_PRIMARY, foreground=self.TEXT_COLOR)
        style.configure('TButton', background=ACCENT, foreground=self.TEXT_COLOR, borderwidth=0)
        style.map('TButton', background=[('active', SELECT_BG), ('pressed', SELECT_BG)])
        style.configure('TRadiobutton', background=BG_PRIMARY, foreground=self.TEXT_COLOR)
        style.map('TRadiobutton', background=[('active', ACCENT)], indicatorcolor=[('selected', SELECT_BG)])
        style.configure('TCheckbutton', background=BG_PRIMARY, foreground=self.TEXT_COLOR)
        style.map('TCheckbutton', background=[('active', ACCENT)], indicatorcolor=[('selected', SELECT_BG)])
        style.configure('TLabelFrame', background=BG_PRIMARY, borderwidth=1, relief=tk.SOLID)
        style.configure('TLabelFrame.Label', background=BG_PRIMARY, foreground=self.TEXT_COLOR)
        style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'), background=SELECT_BG)
        style.map('Accent.TButton', background=[('active', ACCENT), ('pressed', ACCENT)])

    def create_widgets(self):
        # Create main paned window for two columns
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ### MODIFIED ###
        # Create a container frame for the left side that will hold the canvas and scrollbar
        left_container = ttk.Frame(main_paned)
        main_paned.add(left_container, weight=2)

        ### NEW ###
        # Create a Canvas
        canvas = tk.Canvas(left_container, bg="#0d1b2a", highlightthickness=0)
        
        # Create a Scrollbar
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        
        # Create the frame that will contain the widgets and be scrolled
        scrollable_frame = ttk.Frame(canvas)

        # Configure the canvas to use the scrollbar
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # This is the magic that makes the frame scrollable
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Update the scrollregion of the canvas whenever the size of the scrollable_frame changes
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Bind mouse wheel to scroll the canvas
        def _on_mouse_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mouse_wheel)


        # Pack the scrollbar and canvas into the left container
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Right frame for console
        right_frame = ttk.Frame(main_paned, padding=10)
        main_paned.add(right_frame, weight=3)  # Console gets more space
        
        ### MODIFIED ###
        # --- All widgets are now placed inside the 'scrollable_frame' ---
        left_frame = scrollable_frame # for simplicity in refactoring below code
        
        header_label = ttk.Label(left_frame, text="A.N.A", font=("Segoe UI", 24, "bold"))
        header_label.pack(pady=(0, 5), padx=10)
        sub_header_label = ttk.Label(left_frame, text="Automated Neural Adapter", font=("Segoe UI", 12))
        sub_header_label.pack(pady=(0, 20), padx=10)

        # Add GPU info display
        gpu_info_frame = ttk.Frame(left_frame)
        gpu_info_frame.pack(fill=tk.X, pady=5, padx=10)
        gpu_info = "🔵 GPU Available" if torch.cuda.is_available() else "🔴 GPU Not Available"
        if torch.cuda.is_available():
            gpu_info += f" - {torch.cuda.get_device_name(0)}"
        gpu_label = ttk.Label(gpu_info_frame, text=gpu_info, font=("Segoe UI", 10))
        gpu_label.pack()

        self._create_model_output_section(left_frame)
        self._create_dataset_section(left_frame)
        self._create_lora_section(left_frame)
        self._create_training_section(left_frame)

        self.start_button = ttk.Button(left_frame, text="Start Training & Merge", command=self.start_training_thread, style='Accent.TButton')
        self.start_button.pack(fill=tk.X, pady=20, ipady=5, padx=10)
        # --- End of widgets in scrollable_frame ---

        # Console in the right frame
        console_frame = ttk.LabelFrame(right_frame, text="Console Log", padding=10)
        console_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add console controls
        console_controls = ttk.Frame(console_frame)
        console_controls.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(console_controls, text="Clear Console", command=self.clear_console).pack(side=tk.LEFT)
        ttk.Button(console_controls, text="Save Log", command=self.save_console_log).pack(side=tk.LEFT, padx=(5, 0))
        
        self.console_output = tk.Text(
            console_frame, 
            wrap=tk.WORD, 
            height=30, 
            state=tk.DISABLED, 
            relief=tk.FLAT, 
            bg="#1b263b", 
            fg="#e0e1dd", 
            insertbackground=self.TEXT_COLOR,
            font=("Consolas", 10)  # Monospace font for better readability
        )
        
        # Add scrollbar to console
        console_scrollbar = ttk.Scrollbar(console_frame, orient=tk.VERTICAL, command=self.console_output.yview)
        self.console_output.configure(yscrollcommand=console_scrollbar.set)
        
        self.console_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        console_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def clear_console(self):
        self.console_output.config(state=tk.NORMAL)
        self.console_output.delete(1.0, tk.END)
        self.console_output.config(state=tk.DISABLED)

    def save_console_log(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.console_output.get(1.0, tk.END))
                print(f" Log saved to: {file_path}")
            except Exception as e:
                print(f" Error saving log: {e}")

    def _create_model_output_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Model & Output", padding=15)
        frame.pack(fill=tk.X, pady=5, padx=10) ### MODIFIED ###: Added padx
        self._create_input_row(frame, 0, "Base Model ID:", self.model_id)
        self._create_input_row(frame, 1, "Hugging Face Token (optional):", self.hf_token, show='*')
        self._create_browse_row(frame, 2, "Output Directory:", self.output_path, self.select_output_directory)

        # NEW: Add checkbox and dynamic folder selection for disk offload
        offload_check = ttk.Checkbutton(frame, text="Enable Disk Offload (for very large models)", variable=self.enable_disk_offload, command=self._toggle_offload_widgets)
        offload_check.grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=(8,0))
        self.offload_widgets = self._create_browse_row(frame, 4, "Offload Folder:", self.offload_folder_path, self.select_offload_directory)
        
        self._toggle_offload_widgets()  # Set initial visibility

    def _create_dataset_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Dataset Configuration", padding=15)
        frame.pack(fill=tk.X, pady=5, padx=10) ### MODIFIED ###: Added padx
        source_frame = ttk.Frame(frame)
        source_frame.grid(row=0, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Label(source_frame, text="Source:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(source_frame, text="Local File", variable=self.dataset_source, value="local", command=self._toggle_dataset_fields).pack(side=tk.LEFT)
        ttk.Radiobutton(source_frame, text="Hugging Face Hub", variable=self.dataset_source, value="hub", command=self._toggle_dataset_fields).pack(side=tk.LEFT, padx=10)
        self.local_widgets = self._create_browse_row(frame, 1, "Local Dataset Path:", self.local_dataset_path, self.open_file_dialog)
        self.hf_id_widgets = self._create_input_row(frame, 2, "HF Dataset ID:", self.hf_dataset_id)
        self.hf_split_widgets = self._create_input_row(frame, 3, "HF Dataset Split:", self.hf_dataset_split)
        self._toggle_dataset_fields()

    def _create_lora_section(self, parent):
        frame = ttk.LabelFrame(parent, text="LoRA Hyperparameters", padding=15)
        frame.pack(fill=tk.X, pady=5, padx=10) ### MODIFIED ###: Added padx
        self._create_input_row(frame, 0, "LoRA R (Rank):", self.lora_r)
        self._create_input_row(frame, 1, "LoRA Alpha:", self.lora_alpha)
        self._create_input_row(frame, 2, "LoRA Dropout:", self.lora_dropout)
        self._create_input_row(frame, 3, "Target Modules (comma-separated):", self.target_modules)
        
    def _create_training_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Training Hyperparameters", padding=15)
        frame.pack(fill=tk.X, pady=5, padx=5) ### MODIFIED ###: Added padx
        self._create_input_row(frame, 0, "Epochs:", self.epochs)
        self._create_input_row(frame, 1, "Max Sequence Length:", self.max_length)
        self._create_input_row(frame, 2, "Batch Size:", self.batch_size)
        self._create_input_row(frame, 3, "Learning Rate:", self.learning_rate)
        optimizer_check = ttk.Checkbutton(frame, text="Use Paged Optimizer (Saves VRAM)", variable=self.use_paged_optimizer)
        optimizer_check.grid(row=4, column=0, columnspan=3, sticky="w", padx=5, pady=8)

    def _create_input_row(self, parent, row, label_text, variable, show=None):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        entry = ttk.Entry(parent, textvariable=variable, show=show)
        entry.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        parent.grid_columnconfigure(1, weight=1)
        return (label, entry)

    def _create_browse_row(self, parent, row, label_text, variable, command):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        button = ttk.Button(parent, text="Browse...", command=command)
        button.grid(row=row, column=2, sticky="e", padx=5, pady=5)
        parent.grid_columnconfigure(1, weight=1)
        return (label, entry, button)

    # NEW: Method to show/hide the offload folder widgets
    def _toggle_offload_widgets(self):
        if self.enable_disk_offload.get():
            for widget in self.offload_widgets:
                widget.grid()
        else:
            for widget in self.offload_widgets:
                widget.grid_remove()

    def _toggle_dataset_fields(self):
        source = self.dataset_source.get()
        if source == "local":
            # Show local widgets, hide HF widgets
            for widget in self.local_widgets:
                widget.grid()
            for widget in self.hf_id_widgets:
                widget.grid_remove()
            for widget in self.hf_split_widgets:
                widget.grid_remove()
        else:
            # Hide local widgets, show HF widgets
            for widget in self.local_widgets:
                widget.grid_remove()
            for widget in self.hf_id_widgets:
                widget.grid()
            for widget in self.hf_split_widgets:
                widget.grid()

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(title="Select Dataset File", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path: 
            self.local_dataset_path.set(file_path)

    def select_output_directory(self):
        folder = filedialog.askdirectory(title="Select Output Directory")
        if folder: 
            self.output_path.set(folder)

    # NEW: Method to select the offload directory
    def select_offload_directory(self):
        folder = filedialog.askdirectory(title="Select Offload Directory")
        if folder: 
            self.offload_folder_path.set(folder)

    def start_training_thread(self):
        self.start_button.config(state=tk.DISABLED, text="Training in Progress...")
        threading.Thread(target=self._run_training, daemon=True).start()

    def _run_training(self):
        original_stdout = sys.stdout
        sys.stdout = ConsoleRedirector(self.console_output, original_stdout)
        try:
            # NEW: Get the offload path only if the checkbox is enabled
            offload_path = None
            if self.enable_disk_offload.get():
                path = self.offload_folder_path.get().strip()
                if not path:
                    raise ValueError("Disk Offload is enabled, but no offload folder was selected.")
                offload_path = path
            
            # Print start time and parameters
            start_time = datetime.now()
            print(f" Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(" Training Parameters:")
            print(f"   - Model: {self.model_id.get()}")
            print(f"   - Epochs: {self.epochs.get()}")
            print(f"   - Batch Size: {self.batch_size.get()}")
            print(f"   - Learning Rate: {self.learning_rate.get()}")
            print(f"   - LoRA Rank: {self.lora_r.get()}")
            print(f"   - LoRA Alpha: {self.lora_alpha.get()}")
            print("-" * 50)
            
            train_params = {
                "model_id": self.model_id.get(),
                "merged_output_path": self.output_path.get(),
                "epochs": self.epochs.get(),
                "lora_r": self.lora_r.get(),
                "lora_alpha": self.lora_alpha.get(),
                "lora_dropout": self.lora_dropout.get(),
                "target_modules": self.target_modules.get(),
                "dataset_source": self.dataset_source.get(),
                "dataset_path": self.local_dataset_path.get(),
                "hf_dataset_id": self.hf_dataset_id.get(),
                "hf_dataset_split": self.hf_dataset_split.get(),
                "max_length": self.max_length.get(),
                "batch_size": self.batch_size.get(),
                "learning_rate": self.learning_rate.get(),
                "hf_token": self.hf_token.get().strip(),
                "use_paged_optimizer": self.use_paged_optimizer.get(),
                "offload_folder": offload_path,  # Pass the path to the training function
                "logger_callback": print,
            }
            train_and_merge(**train_params)
            
            # Print completion time and duration
            end_time = datetime.now()
            duration = end_time - start_time
            print(f" Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱  Total duration: {duration}")
            
            self.show_message_box("Training Complete", "Model training and merging finished successfully!")
        except Exception as e:
            print(f" An error occurred during training:\n{'-'*30}")
            traceback.print_exc()
            print(f"{'-'*30}")
            self.show_message_box("Training Error", f"An error occurred: {e}", is_error=True)
        finally:
            sys.stdout = original_stdout
            self.root.after(0, self._reset_button_state)

    def _reset_button_state(self):
        self.start_button.config(state=tk.NORMAL, text="Start Training & Merge")

    def show_message_box(self, title: str, message: str, is_error: bool = False):
        if is_error:
            self.root.after(0, lambda: messagebox.showerror(title, message))
        else:
            self.root.after(0, lambda: messagebox.showinfo(title, message))


if __name__ == "__main__":
    root = tk.Tk()
    app = LoRATrainerApp(root)
    root.mainloop()



def run():
    root = tk.Tk()
    app = LoRATrainerApp(root)
    root.mainloop()