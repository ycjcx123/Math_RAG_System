from src import MathEvaluator, Score

from src.utils.config_loader import load_config


if __name__ == '__main__':
    config=load_config()
    evaluator = MathEvaluator(config)
    score = Score(config)

    evaluator.run_all_tasks()
    score.run_evaluation()