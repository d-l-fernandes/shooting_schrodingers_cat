from absl import app

import trainers


def main(argv):
    trainer = trainers.ModelTrainer()
    trainer.run()


if __name__ == "__main__":
    app.run(main)
