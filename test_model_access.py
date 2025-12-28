import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ================= CONFIGURATION =================
# 1. The exact folder name as seen in your screenshot
TARGET_MODEL_NAME = "Qwen2-7B-Instruct"

# 2. OPTION A: The Full Absolute Path (EDIT THIS)
#    Example: "/home/administrator/models/Qwen2-7B-Instruct"
TEST_ABSOLUTE_PATH = "/opt/pretrained_models/" + TARGET_MODEL_NAME

# 3. OPTION B: The Environment Variable Name (if you set one in .bashrc)
ENV_VAR_NAME = "MODEL_DIR_PATH"
# =================================================


def print_status(method, status, message):
    color = "\033[92m" if status == "SUCCESS" else "\033[91m"
    reset = "\033[0m"
    print(f"[{method}] {color}{status}{reset}: {message}")


def try_load_model(path_to_test, method_name):
    print(f"\n--- Testing Method: {method_name} ---")
    print(f"Looking at path: {path_to_test}")

    # 1. Check if path string is valid
    if not path_to_test or not isinstance(path_to_test, str):
        print_status(method_name, "SKIPPED", "Path is empty or invalid.")
        return False

    # 2. Check if directory exists on disk
    if not os.path.exists(path_to_test):
        print_status(method_name, "FAILURE", "Directory not found on disk.")
        return False

    # 3. Check for essential files (config.json)
    if not os.path.exists(os.path.join(path_to_test, "config.json")):
        print_status(
            method_name,
            "FAILURE",
            "Directory exists, but 'config.json' is missing. Is this a valid model folder?",
        )
        return False

    # 4. Attempt to load Config (Lightweight check)
    try:
        config = AutoConfig.from_pretrained(path_to_test, trust_remote_code=True)
        print("   > Config loaded successfully.")
    except Exception as e:
        print_status(method_name, "FAILURE", f"Crashed loading config: {e}")
        return False

    # 5. Attempt to load Tokenizer (Medium check)
    try:
        tokenizer = AutoTokenizer.from_pretrained(path_to_test, trust_remote_code=True)
        print("   > Tokenizer loaded successfully.")
    except Exception as e:
        print_status(method_name, "FAILURE", f"Crashed loading tokenizer: {e}")
        return False

    # 6. Attempt to load Weights to GPU (Heavy check)
    #    We check for GPU availability first
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   > Attempting to load weights onto {device}...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            path_to_test,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,  # Using float16 for speed/memory safety during test
        )
        print_status(
            method_name,
            "SUCCESS",
            f"Model loaded! VRAM used: {model.get_memory_footprint() / 1e9:.2f} GB",
        )

        # Cleanup to free memory for next test
        del model
        del tokenizer
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print_status(method_name, "FAILURE", f"Crashed loading weights: {e}")
        return False


def main():
    print("=====================================================")
    print(
        f"Checking GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU Detected'}"
    )
    print("=====================================================")

    # TEST 1: Absolute Path
    # This is the most reliable method. If this fails, the path is wrong.
    try_load_model(TEST_ABSOLUTE_PATH, "Absolute Path")

    # # TEST 2: Environment Variable
    # # This tests if your .bashrc export is working and accessible by Python.
    # env_path = os.environ.get(ENV_VAR_NAME)
    # if env_path:
    #     full_env_path = os.path.join(env_path, TARGET_MODEL_NAME)
    #     try_load_model(full_env_path, "Environment Variable")
    # else:
    #     print_status(
    #         "Environment Variable",
    #         "SKIPPED",
    #         f"Variable '{ENV_VAR_NAME}' is not set in this shell.",
    #     )

    # TEST 3: Direct Short Name (The specific test you requested)
    # This checks: Can I just write "Qwen2-7B-Instruct" and have it work?
    # Expected result: FAILURE unless you are running this script inside the 'models' folder.
    try_load_model(TARGET_MODEL_NAME, "Direct Short Name")


if __name__ == "__main__":
    main()
