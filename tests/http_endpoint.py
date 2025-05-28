import tensorflow_estimator as tf_estimator
from addressnet.model import model_fn


def create_estimator(model_dir):
    return tf_estimator.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)


if __name__ == "__main__":
    # Simple runtime check that estimator can be created
    import tempfile
    est = create_estimator(tempfile.mkdtemp())
    print("Created estimator", type(est))

