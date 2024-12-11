import pandas as pd
import geopandas as gpd


def processing_train_data():
    train_df = pd.read_csv("./train/data/train_small.csv")
    boundary_code = pd.read_excel(
        "./geodata/AdminiBoundary_CD.xlsx", header=1, skiprows=[0]
    )

    boundary_code_dict = {}

    # 行政区域コードの処理
    for row in boundary_code.itertuples():
        # 改正後の名称が存在しない場合(nan)
        if type(row[9]) is float:
            bnd_code = row[1]
            # 都道府県名しか存在しない場合
            if type(row[3]) is float:
                short_address = row[2]
            else:
                short_address = row[2] + row[3]
        # 改正後の名称が存在する場合(str)
        else:
            bnd_code = row[8]  # 改正後の行政区域コード
            short_address = row[2] + row[9]

        boundary_code_dict[short_address] = bnd_code

    # 訓練データの処理
    processed_address = train_df["full_address"].str.extract(r"^(.*?(?:町|市|区|府))")

    code_column = []

    for row in processed_address.itertuples():
        if row[1] in boundary_code_dict:
            code_column.append(boundary_code_dict[row[1]])
        else:
            code_column.append(None)

    train_df["boundary_code"] = pd.Series(code_column)
    train_df = train_df.dropna(subset=["boundary_code"])

    return train_df


def main():
    enc = "shift-jis"
    gdf = gpd.read_file("./geodata/L01-23_GML/L01-23.shp", encoding=enc)
    df = processing_train_data()

    df["boundary_code"] = df["boundary_code"].astype(int)
    gdf_necessary = gdf[['L01_022', 'L01_006']]
    gdf_necessary.columns = ['boundary_code', 'price']
    gdf_necessary["boundary_code"] = gdf_necessary["boundary_code"].astype(int)
    merged_df = gdf_necessary.merge(df, on="boundary_code")

    merged_df.to_csv("output.csv", index=False)


if __name__ == "__main__":
    main()
