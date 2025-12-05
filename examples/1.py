from core.runner import AutoMLBench


def main():
    bench = AutoMLBench(validation_metric='f1', test_metrics=['average_precision', 'mcc'])
    bench.run()


if __name__ == '__main__':
    main()