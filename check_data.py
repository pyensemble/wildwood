from wildwood.datasets._adult import _fetch_adult, load_adult


dataset = load_adult()

print(dataset)

print(dataset.df_raw)

X_train, X_test, y_train, y_test = dataset.extract(42)

print(dataset.columns_)
