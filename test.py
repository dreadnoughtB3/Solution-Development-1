import pandas as pd
import re


def truncate_address(addresses):
    result = []
    for address in addresses:
        # パターンの優先順: 町, 区, 市, 県/府
        match = re.search(r'(.*?\b(町|区|市|府|県)\b)', address)
        if match:
            result.append(match.group(1))
        else:
            result.append(address)
    return result


train_df = pd.read_csv("./train/data/train_small.csv")
boundry_code = pd.read_excel("./geodata/AdminiBoundary_CD.xlsx", header=1, skiprows=[0])

boundry_code["full_address"] = boundry_code["都道府県名"] + boundry_code["市区町村名"]
train_df['truncated_address'] = truncate_address(train_df["full_address"])
merged_data = pd.merge(
    train_df,
    boundry_code,
    left_on="truncated_address",
    right_on="full_address",
    how='left'
)

merged_data.to_csv('output.csv', index=False)
