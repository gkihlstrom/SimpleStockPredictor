from classes import ModelFactory

if __name__ == "__main__":
    mf = ModelFactory()
    mf.create_model()
    mf.oos_test()
