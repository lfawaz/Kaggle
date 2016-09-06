import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object KaggleAllStateChallenge {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Kaggle All State Challenge").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val train = sc.textFile("../data/train.csv")
    print(train.first())
   
  }
}