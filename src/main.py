from src.utils import *

df = load_data()
df = remove_direct_entries(df, 'medium')
df = remove_outliers_z_score(df, 3.5)
