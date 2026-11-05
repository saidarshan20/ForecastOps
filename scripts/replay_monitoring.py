from forecastops.monitoring.replay import run


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    summary = run(args.database, summary_path=args.summary)
    print(f"Replay: {summary['paired_days']} paired days, selected MAE {summary['selected_day_ahead_mae']:.2f}")
