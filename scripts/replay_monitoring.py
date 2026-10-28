from forecastops.monitoring.replay import run


if __name__ == "__main__":
    summary = run()
    print(f"Replay: {summary['paired_days']} paired days, selected MAE {summary['selected_day_ahead_mae']:.2f}")
