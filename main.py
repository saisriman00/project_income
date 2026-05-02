from src.data_ingestion import ingest_data
from src.data_split import split_data
from pipelines.data_processing_pipeline import process_data
from pipelines.training_pipeline import train_models_pipeline
from pipelines.evaluation_pipeline import evaluate_pipeline


def main():
    print("Step 1: Data Ingestion...")
    ingest_data()

    print("Step 2: Data Splitting...")
    split_data()

    print("Step 3: Data Processing...")
    process_data()

    print("Step 4: Model Training...")
    train_models_pipeline()

    print("Step 5: Model Evaluation...")
    best_model = evaluate_pipeline()

    print(f"Best Model: {best_model}")


if __name__ == "__main__":
    main()
