import org.apache.spark._
//import org.apache.spark.SparkConf
//import org.apache.spark.SparkContext


  val conf = new SparkConf().set("spark.driver.allowMultipleContexts", "true")
    .setAppName("test")
      .setMaster("local[*]")

  val sc = new SparkContext(conf)

val lines = sc.textFile("Data/train.csv")

lines.first()