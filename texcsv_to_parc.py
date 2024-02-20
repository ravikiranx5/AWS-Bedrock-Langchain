from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder.appName("CSVToParquet").getOrCreate()

# Read the CSV file into a DataFrame
csv_df = spark.read.option("header", "true").csv("C:\\Users\\ravik\\Downloads\\parquet samples\\karma\\cat_travelworld.csv")

# Write the DataFrame to a Parquet file
csv_df.write.parquet("C:\\Users\\ravik\\Downloads\\parquet samples\\karma\\cat_travelworld.parquet")

# Stop SparkSession
spark.stop()
