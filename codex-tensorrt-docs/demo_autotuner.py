from tensorrt_llm._torch.autotuner import AutoTuner, autotune


def main():
    tuner = AutoTuner.get()
    tuner.clear_cache()
    tuner.reset_statistics()
    # Place any calls that internally use AutoTuner.choose_one(...) inside autotune()
    with autotune(cache_path="codex-tensorrt-docs/autotune_cache.json"):
        pass
    print(tuner.stats)


if __name__ == "__main__":
    main()

