import sys
import math

def main():
    # Check if chunk_num is provided
    if len(sys.argv) < 2:
        print("Usage: python cluster_size_calculate.py <chunk_num>")
        sys.exit(1)

    try:
        chunk_num = int(sys.argv[1])
    except ValueError:
        print("Error: chunk_num must be an integer.")
        sys.exit(1)

    print(f"\n--- Stage 1: Cluster Calculation for chunk_num (N) = {chunk_num} ---")

    # 1. Calculate min/max clusters
    # Formula: min_cluster = 4 * sqrt(N)
    min_cluster = 4 * math.sqrt(chunk_num)
    
    # Formula: max_cluster = 16 * sqrt(N)
    max_cluster = 16 * math.sqrt(chunk_num)

    # 2. Calculate cluster sizes
    # Size when using min_cluster
    size_at_min = chunk_num / min_cluster
    
    # Size when using max_cluster
    size_at_max = chunk_num / max_cluster

    print(f"Min Clusters (4*sqrt(N)): {min_cluster:.4f}")
    print(f"Max Clusters (16*sqrt(N)): {max_cluster:.4f}")
    print(f"Cluster Size at Min Clusters: {size_at_min:.4f}")
    print(f"Cluster Size at Max Clusters: {size_at_max:.4f}")
    print("-" * 60)

    # Stage 2: Search Range Percentage Computation
    print("\n--- Stage 2: Search Range Percentage Computation ---")
    
    # Get the base number of clusters for the percentage calculation
    while True:
        try:
            choice_input = input("Enter your choice for the number of clusters (or 'quit' to exit): ")
            if choice_input.lower() == 'quit':
                return
            chosen_clusters = float(choice_input)
            if chosen_clusters == 0:
                print("Number of clusters cannot be zero.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    print(f"Base cluster count set to: {chosen_clusters}")
    print("Enter numbers to calculate their percentage relative to the cluster count.")

    # Loop round after round
    while True:
        user_input = input("\nEnter a number (or 'quit' to exit): ")
        
        if user_input.lower() == 'quit':
            break
        
        try:
            number = float(user_input)
            percentage = (number / chosen_clusters)
            # Displaying as a raw ratio and a percentage string
            print(f"Fraction: {percentage:.6f}")
            print(f"Percentage: {percentage * 100:.4f}%")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

if __name__ == "__main__":
    main()
