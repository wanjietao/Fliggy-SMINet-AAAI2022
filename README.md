# Data Source

https://tianchi.aliyun.com/dataset/dataDetail?dataId=113649

# Data Format

### user_item_behavior_history.csv
The data set covers all behaviors (including clicks, favorites, adding, and purchases) of approximately 2 million random users from June 3, 2019 to June 3, 2021. The organization of the data set is similar to MovieLens-20M, namely Each row of the collection represents a user data behavior, which is composed of user ID, product ID, behavior type, and log. Each column summarizes the detailed information of the product under investigation.
| Field | Explanation |
| --- | --- |
| User ID | An integer, the serialized ID that represents a user |
| Item ID | An integer, the serialized ID that represents an item |
| Behavior type | A string, enum-type from ('clk', 'fav', 'cart','pay') |
| Timestamp | An integer, the timestamp of the behavior |


### user_profile.csv
This data set mainly contains the basic attributes of about five million random users, such as age, gender, occupation, resident city ID, crowd tag, etc. Each row of the data set represents a piece of user information, separated by commas. The detailed description of each column in the data set is as follows:
| Field | Explanation |
| --- | --- |
| User ID | An integer, the serialized ID that represents a user |
| Age | An integer, the serialized ID that represents an user age |
| Gender | An integer, the serialized ID that represents the user gender |
| Occupation | An integer, the serialized ID that represents the user occupation |
| Habitual City | An integer, the serialized ID that represents the user habitual city，single value |
| User Tag List | A String, the ID of each tag is separated by English semicolon after serialization |

### item_profile.csv
This data file mainly contains the basic attribute characteristics of about 270,000 products, such as product category ID, product city, product label, etc. Each row of the data set represents the information of a product, separated by commas. The detailed description of each column in the data set is as follows:
| Field | Explanation |
| --- | --- |
| Item ID | An integer, the serialized ID that represents an item |
| Category ID | An integer, the serialized ID that represents an item category |
| Item City | An integer, the serialized ID that represents an item City |
| Item Tag List | A String, the ID of each tag is separated by English semicolon after serialization |
