<h2>IP-ID-Mapping</h2>


<h3>Test data Location : </h3>  

   * hdfs:///temp/nuru/New_Nat.csv
   * hdfs:///temp/nuru/New_Ipdr.csv


<h3>To Run: </h3>
run the command "sbt assembly" to get the .jar file and then run spark submit as follow:
    
    spark-submit --class IP_ID_Mapping \
    --master yarn \
    --deploy-mode cluster \
    --driver-memory 4g \
    --executor-memory 20g \
    --executor-cores 4 \
    --num-executors 4 jar_file \
    "hdfs:///temp/nuru/" \
    "New_Nat.csv" \
    "New_Ipdr.csv" \
    "IP_ID_Mapping_results/"
