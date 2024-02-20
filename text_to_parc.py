from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder.appName("TextToParquet").getOrCreate()

# Read the text file into a DataFrame
text_df = spark.read.text("C:\\Users\\ravik\\Downloads\\parquet samples\\karma\\cat_travelworld.txt")

# Write the DataFrame to a Parquet file
text_df.write.parquet("C:\\Users\\ravik\\Downloads\\parquet samples\\karma\\cat_travelworld.parquet")

# Stop SparkSession
spark.stop()
