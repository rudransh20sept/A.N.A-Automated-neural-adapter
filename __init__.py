from .train import train
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run LoRA fine-tuning and merging.")
    parser.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Hugging Face model ID")
    parser.add_argument("--merged_output_path", type=str, default="merged_model_output", help="Output directory for merged model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj", help="Comma-separated list of target modules")
    parser.add_argument("--dataset_source", type=str, choices=["local", "hub"], default="local", help="Dataset source: local or hub")
    parser.add_argument("--dataset_path", type=str, default="data/dummy_dataset.txt", help="Path to local dataset file")
    parser.add_argument("--hf_dataset_id", type=str, default="HuggingFaceH4/ultrachat_200k", help="Hugging Face dataset ID")
    parser.add_argument("--hf_dataset_split", type=str, default="train_sft", help="Hugging Face dataset split")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")

    args = parser.parse_args()

    # ✅ Just call train() directly with arguments
    train(
        model_id=args.model_id,
        merged_output_path=args.merged_output_path,
        epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        dataset_source=args.dataset_source,
        dataset_path=args.dataset_path,
        hf_dataset_id=args.hf_dataset_id,
        hf_dataset_split=args.hf_dataset_split,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logger_callback=print,
    )

if __name__ == "__main__":
    main()
