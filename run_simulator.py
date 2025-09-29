from simulator import run_simulation

if __name__ == "__main__":
    df, assignment_log = run_simulation(num_nodes=10, timesteps=80, initial_train_end=50, verbose_steps=12)
    print("\nSimulation completed. Results saved as:")
    print(" - edge_dataset.csv")
    print(" - edge_dataset.pdf")
    print(" - edge_dataset_after_assignments.csv")
    print(" - assignment_log.csv")
    print("\nLast few assignments:")
    print(assignment_log.tail(10))