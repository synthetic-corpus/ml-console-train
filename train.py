import argparse


def main():
    parser = argparse.ArgumentParser(description="Training script for ML console.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'frames' subcommand
    frames_parser = subparsers.add_parser("frames", help="Work with frames data")
    frames_parser.add_argument(
        "--sql-only",
        action="store_true",
        help="Only tests connection to SQL."
    )

    # 'train' subcommand
    train_parser = subparsers.add_parser("train", help="Train a machine learning model")
    train_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["forest", "regression", "vector"],
        help="Specify which model to train: 'forest' \
            (Random Forest Classifier), 'regression' (Logistic Regression), \
                or 'vector' (Support Vector Machine)."
    )
    # The --model argument determines which ML model to use:
    #   'forest'    -> Random Forest Classifier
    #   'regression'-> Logistic Regression
    #   'vector'    -> Support Vector Machine

    args = parser.parse_args()

    if args.command == "frames":
        if args.sql_only:
            # Placeholder for --sql-only logic
            pass
        else:
            # Placeholder for frames logic
            pass
    elif args.command == "train":
        # Placeholder for training logic based on args.model
        pass

if __name__ == "__main__":
    main() 