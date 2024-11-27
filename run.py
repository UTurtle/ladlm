import subprocess

def main():
    # print("=== Step 1: Download Dataset ===")
    # subprocess.run(["python", "get_dataset.py"], check=True)

    print("\n=== Step 2: Preprocess Dataset ===")
    subprocess.run(["python", "preprocessing/baseline.py"], check=True)

    print("\n=== Step 3: Fine-tune Model ===")
    subprocess.run(["python", "peft/baseline.py"], check=True)

    print("\n=== Step 4: Test Fine-tuned Model ===")
    subprocess.run(["python", "train/baseline.py"], check=True)

    print("\nAll steps completed successfully!")

if __name__ == "__main__":
    main()
