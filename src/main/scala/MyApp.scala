package upm.bd
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.ml._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import java.time.LocalDate 
import java.time.format.DateTimeFormatter
import java.time.YearMonth
import java.time.temporal.ChronoUnit.DAYS
import org.apache.log4j._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.param.shared.HasHandleInvalid
import scala.io.Source
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}

object MyApp extends remainingMethodsClass{

    def main(args : Array[String]) {
        Logger.getLogger("org").setLevel(Level.ERROR)
        val conf = new SparkConf().setAppName("My first Spark application")
        val sc = new SparkContext(conf)
            .setCheckpointDir("/tmp/checkpoints")
        val spark = SparkSession.builder()
            .appName("Spark SQL example")
            .getOrCreate()

        // read the input file
        val auxOutput = readInput(args, spark)
        var rawDF = auxOutput._1
        val wrongColumnsList = auxOutput._2
        // val machineLearningAlg = auxOutput._3
        val machineLearningAlg = "lr"
        // throw error if not all columns are present or if file is empty
        checkIfEmpty(rawDF)        
        rawDF = rawDF.distinct
        val dropperTransformer = new dropper().setHandleInvalid("skip") // drop all forbidden/useless columns
        dropperTransformer.wrongColumnsList = wrongColumnsList // adding the columns that really should be removed
        val emptyColumnsAndFieldsTransformer = new emptyColumnsAndFields().setHandleInvalid("skip") // get all string columns && remove all remaining null fields OR drop column with too many NAs
        val formattedDateTransformer = new formattedDate().setHandleInvalid("skip")  // add column for further analysis (proximityToHolidayTransformer   )
        val stringTypesToIntTransformer = new stringTypesToInt().setHandleInvalid("skip") // cast possible string columns to integer
        val countriesAndCarriersAndFlightNumToNumbersTransformer = new countriesAndCarriersAndFlightNumToNumbers().setHandleInvalid("skip") // here we collect all the countries/carrierCodes present in origin/dest/codes & change strings to numbers
        val proximityToHolidayTransformer = new proximityToHoliday().setHandleInvalid("skip") // here we compute the US holiday/event list for those Year values (holidayCalendar class defined below)
        var pipelineStages : Array[Transformer] = Array(
            dropperTransformer,
            emptyColumnsAndFieldsTransformer,
            formattedDateTransformer,
            stringTypesToIntTransformer,
            countriesAndCarriersAndFlightNumToNumbersTransformer,
            proximityToHolidayTransformer
        )

        // if (machineLearningAlg.toLowerCase == "lr"){
        //     val depDelayTaxiOutInteractionTransformer = new depDelayTaxiOutInteraction().setHandleInvalid("skip") // create intraction variable between old DepDelay and TaxiOut
        //     val origDestInteractionTransformer = new origDestInteraction().setHandleInvalid("skip") // create intraction variable between Origin and Dest
        //     val monthDayofMonthInteractionTransformer = new  monthDayofMonthInteraction().setHandleInvalid("skip") // create interaction variable between Month and DayofMonth
        //     val depDelayOrigInteractionTransformer = new depDelayOrigInteraction().setHandleInvalid("skip")
        //     val depDelayCRSArrTimegInteractionTransformer = new depDelayCRSArrTimegInteraction().setHandleInvalid("skip")
        // }

        val dropRemainingStringTransformer = new dropRemainingString() // just drop anything that was not succesfully converted to Int
        pipelineStages = pipelineStages :+ dropRemainingStringTransformer
        val pipelineTransformers = new Pipeline().setStages(
            pipelineStages
        )
        // transform the original DF into the DF we are going to use
        val intermediaryPipeline = pipelineTransformers.fit(rawDF)
        var intermediaryDF = intermediaryPipeline.transform(rawDF)
        // get column names that are not ArrDelay
        var variableNames = intermediaryDF.columns.filter(_ != "ArrDelay")

        if (machineLearningAlg == "lr")
            for (colName <- variableNames   ){
                var new_name = colName + "Squared"
                intermediaryDF = intermediaryDF.withColumn(new_name, col(colName) * col(colName))
            }

        // create the assembler that's gonna add the features column
        val generalAssembler = new VectorAssembler().setInputCols(variableNames).setOutputCol("features").setHandleInvalid("skip")
        // create the pipeline that's gonna use the assembler
        val pipelineAssembler = new Pipeline().setStages(Array(generalAssembler))
        // creating the new DF
        val pipelineAssemblerRes = pipelineAssembler.fit(intermediaryDF)
        val featuresDF = pipelineAssemblerRes.transform(intermediaryDF)
        // split the data
        val splitData = featuresDF.randomSplit(Array(0.8, 0.2))

        // get training and test
        var training = splitData(0)
        var test = splitData(1)

        //create the standard caler
        val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(true)

        // create the pipeline for the standard scaler
        val pipelineScaler = new Pipeline().setStages(Array(scaler))
        // get the scaler model
        val scalerModel = pipelineScaler.fit(training)
        // use the same scaler model to transform both test and training so no info is leaked from test to train
        val scaledTraining = scalerModel.transform(training)
        val scaledTest = scalerModel.transform(test)

        val cv = createMLModel(machineLearningAlg)

        // get the best model
        val cvModel = cv.fit(scaledTraining)

        // try the test dataset with the best model found
        val testTransformed = cvModel.transform(scaledTest)

        // create the evaluator with the rmse metric
        val lr_evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("ArrDelay").setPredictionCol("prediction")
        
        // pray to God the results are good
        println(lr_evaluator.evaluate(testTransformed))

        //("C:\\Users\\User\\Desktop\\UPM_2021_2022\\SEMESTER_1\\BIG_DATA\\project\\empty_example.csv" )
    }
}

// this is a function every transformer needs but we were too lazy to write it in every one, so we created a trait
trait columnExisting{
    def columnExists(df: DataFrame, columnName: String) : Boolean = {
        try{
            df(columnName)
            true
        } catch {
            case e: Exception =>
                false
        }
    }
}

class remainingMethodsClass extends columnExisting{

    def readInput(args: Array[String], spark: SparkSession) : (DataFrame, Array[String], String) = {
        // RULE: everything that is not CRScheduled is String.
        if (args.length >= 1) {
            if (args(0).takeRight(4) != ".csv") {
                throw new Exception("File format is not supported. Please use .csv")
            }
            val sourceFile = Source.fromFile(args(0))
            val headerLine = sourceFile.getLines.take(1).toArray
            sourceFile.close
            val headersRaw = headerLine(0).split(",")
            val headersFinal = headersRaw.map(_.replaceAll("\\W", ""))
            val columnTypesHash = Map(
                ("Year", IntegerType),
                ("Month", IntegerType),
                ("DayofMonth", IntegerType),
                ("DayOfWeek", IntegerType),
                ("DepTime", StringType),
                ("CRSDepTime", StringType),
                ("ArrTime", StringType),
                ("CRSArrTime", StringType),
                ("UniqueCarrier", StringType),
                ("FlightNum", IntegerType),
                ("TailNum", StringType),
                ("ActualElapsedTime", StringType),
                ("CRSElapsedTime", IntegerType),
                ("AirTime", StringType),
                ("ArrDelay", StringType),
                ("DepDelay", StringType),
                ("Origin", StringType),
                ("Dest", StringType),
                ("Distance", IntegerType),
                ("TaxiIn", StringType),
                ("TaxiOut", StringType),
                ("Cancelled", IntegerType),
                ("CancellationCode", StringType),
                ("Diverted", IntegerType),
                ("CarrierDelay", StringType),
                ("WeatherDelay", StringType),
                ("NASDelay", StringType),
                ("SecurityDelay", StringType),
                ("LateAircraftDelay", StringType)
            )
            val correctColumnNamesList = columnTypesHash.keys.toArray
            var schemaArray : Array[StructField] = Array()
            var wrongColumnArray : Array[String] = Array()
            for ( columnName <- headersFinal){
                if (correctColumnNamesList.contains(columnName))
                    schemaArray = schemaArray :+ StructField(columnName, columnTypesHash(columnName), true)
                else{
                    schemaArray = schemaArray :+ StructField(columnName, StringType, true)
                    wrongColumnArray = wrongColumnArray :+ columnName
                    println(s"$columnName has a name that is not in the original 29. It will be removed during processing.")
                }
            }
            val genericSchema = StructType(schemaArray)
            var machineLearningAlg = ""
            var extraDataset = ""
            if (args.length >= 2){
                machineLearningAlg = args(1)
            }
            return (spark.read.schema(genericSchema).options(Map("header"->"true")).csv(args(0)), wrongColumnArray, machineLearningAlg)
        } 
        else {
            throw new Exception("locaiton of the input file is missing or incorrect")
        }
    }

    // checks to make sure the input is good :) 
    def checkIfEmpty( df : DataFrame ) : Boolean = {
        try {
            df.first
        }catch{ 
            case e: java.util.NoSuchElementException =>
                throw new Exception("The file is, sadly, empty. Please choose a file containing actual data.")
        }
        return true
    }

    def createMLModel( chosenAlgorithm : String ) : CrossValidator = {
        if (chosenAlgorithm.toLowerCase == "lr") {
            val algorithm = new LinearRegression().setFeaturesCol("features").setLabelCol("ArrDelay").setMaxIter(1000).setElasticNetParam(1)
            val paramGrid = new ParamGridBuilder()
            .addGrid(
                algorithm.regParam, 
                Array(
                    0.1, 0.2, 0.3
                )
            ).addGrid(
                algorithm.elasticNetParam,
                Array(
                    0.5, 0.8, 1
                )
            ).build()
            val pipelineRegression = new Pipeline().setStages(Array(algorithm))
        // add cross validation combined with hyperparameter tuning and choosing the best model
            val cv = new CrossValidator().setEstimator(pipelineRegression).setEvaluator(
                    new RegressionEvaluator()
                    .setMetricName("rmse")
                    .setLabelCol("ArrDelay")
                    .setPredictionCol("prediction")
                ).setEstimatorParamMaps(paramGrid).setNumFolds(3).setParallelism(2)
            return cv
        } else {//if (chosenAlgorithm.toLowerCase == "rf") {
            val algorithm = new RandomForestRegressor().setLabelCol("ArrDelay").setFeaturesCol("scaledFeatures").setNumTrees(11).setMaxDepth(11).setFeatureSubsetStrategy("auto").setImpurity("variance").setMaxBins(100)
            val paramGrid = new ParamGridBuilder()
            .addGrid(
                algorithm.maxDepth, 
                Array(
                    10,12
                )
            ).build()
            val pipelineRegression = new Pipeline().setStages(Array(algorithm))
            // add cross validation combined with hyperparameter tuning and choosing the best model
            val cv = new CrossValidator().setEstimator(pipelineRegression).setEvaluator(
                    new RegressionEvaluator()
                    .setMetricName("rmse")
                    .setLabelCol("ArrDelay")
                    .setPredictionCol("prediction")
                ).setEstimatorParamMaps(paramGrid).setNumFolds(5).setParallelism(2)
            return cv
        } 
    }
 }

// this class is used to add a new feature to the dataframe, considering US holiday dates
class holidayCalendar extends Serializable (){
    private def getFirstDayOfMonth(year : String, month : String) : Integer = {
        val formatGNWOKM = DateTimeFormatter.ofPattern("d-M-y")
        val localDate = LocalDate.parse("01-" + month + "-" + year, formatGNWOKM)
        localDate.getDayOfWeek().getValue()
    }

    private def getNthDayOfMonth(year : String, month : String, nthDay : Integer, day : Integer) : String = {
        val firstDayOfMonth = getFirstDayOfMonth(year, month)
        val date = day - firstDayOfMonth + 1
        var stringDate = "01"
        if (date + (nthDay-1)*7 > YearMonth.of(year.toInt, month.toInt).lengthOfMonth()){
            throw new Exception("impossible to compute sadly :(")
        } else {
            if (date < 0)
                stringDate = (date + (nthDay)*7).toString
            else
                stringDate = (date + (nthDay-1)*7).toString
        }
        stringDate + "-" + month + "-" + year
    }

    def holidaysForYear(year : String) : Array[LocalDate] = {
        val format = DateTimeFormatter.ofPattern("d-M-y")
        val sundanceFilmFestivar = LocalDate.parse("29-01-" + year, format)
        val independenceDay = LocalDate.parse("04-07-" + year, format)
        val usOpenTennis = LocalDate.parse("27-08-" + year, format)
        var halloween  = LocalDate.parse("31-10-" + year, format)
        val newYear = LocalDate.parse("31-12-" + year, format)
        val christmasEve = LocalDate.parse("25-12-" + year, format)
        val veteransDay = LocalDate.parse("11-11-" + year, format)
        val juneteenthDay = LocalDate.parse("19-06-" + year, format)
        val easterApprox = LocalDate.parse("04-04-" + year, format)
        val superBowlSunday = LocalDate.parse(getNthDayOfMonth(year,"02", 1, 7), format) //"1st_sunday_feb"
        val mastersGolfTournament = LocalDate.parse(getNthDayOfMonth(year, "04", 2, 7), format) //"2nd_sunday_of_april"
        val theBostonMarathon = LocalDate.parse(getNthDayOfMonth(year, "04", 3, 1), format) //"3rd_monday_in_april"
        val thanksgivingDay = LocalDate.parse(getNthDayOfMonth(year, "11", 4, 4), format) //"4th_thursday_of_nov"
        val laborDay = LocalDate.parse(getNthDayOfMonth(year, "10", 1, 1), format) //"1st_monday_sept"
        val columbusDay = LocalDate.parse(getNthDayOfMonth(year, "09", 2, 1), format) //"2nd_monday_in_oct"
        val mlkJRDay = LocalDate.parse(getNthDayOfMonth(year, "01", 3, 1), format) // "3rd_monday_in_jan"
        val presidentsDay = LocalDate.parse(getNthDayOfMonth(year, "02", 3, 1), format) //"3rd_monday_feb"
        val burningMan = laborDay //"depends_on_Labor_day"
        val mardiGras =  easterApprox.minusDays(47)//"47_before_easter"
        try{
            val memorialDay = LocalDate.parse(getNthDayOfMonth(year, "05", 5, 1), format)
        }catch {
            case e: Exception => 
                val memorialDay = LocalDate.parse(getNthDayOfMonth(year, "05", 4, 1),format)
        } // "last_monday_in_may"

        Array(sundanceFilmFestivar, independenceDay, usOpenTennis, halloween, 
            newYear, christmasEve, veteransDay, juneteenthDay, easterApprox, superBowlSunday,
            mastersGolfTournament, theBostonMarathon, thanksgivingDay, laborDay, columbusDay,
            mlkJRDay, presidentsDay, burningMan, mardiGras)
    }
}

// drop useless/forbidden columns
class dropper(override val uid: String) extends Transformer with HasHandleInvalid{ 
    var wrongColumnsList : Array[String] = Array() 
    override def copy(extra: ParamMap): dropper = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("dropper"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)
        
        val noWrongColumns = df.drop(wrongColumnsList: _*)
        val noCancelColumnDF = noWrongColumns.filter(col("Cancelled") === 0)
        // try to drop columns
        val featuresToBeDropped = Array(
            "ArrTime", "ActualElapsedTime", "AirTime", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "TaxiIn", // find then drop the forbidden columns
            "TailNum",    // also drop the TailNum column given that it's an ID and it's not useful for the model
            "Cancelled", "CancellationCode" // these are also useless after we deleted rows with missing data
        )
        val usableDF = noCancelColumnDF.drop(featuresToBeDropped: _*)
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}

// remove all remaining null fields OR drop column with too many NAs
class emptyColumnsAndFields(override val uid: String) extends Transformer with HasHandleInvalid{    
    override def copy(extra: ParamMap): emptyColumnsAndFields = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("emptyColumnsAndFields"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        for (columnName <- usableDF.columns){
            val countNA = usableDF.filter(col(columnName) === "NA").count
            val countNULL = usableDF.filter(col(columnName).isNull).count
            val countTotal = usableDF.select(col(columnName)).count

            if (usableDF.schema(columnName).dataType == org.apache.spark.sql.types.StringType)
                if (countNA/countTotal.toFloat >= 0.3)
                    usableDF = usableDF.drop(columnName)
                else{
                    usableDF = usableDF.filter(col(columnName) =!= "NA")
                }
            else
                if (countNULL/countTotal.toFloat >= 0.3)
                    usableDF = usableDF.drop(columnName)
                else{
                    usableDF = usableDF.filter(col(columnName).isNull === false)
                }
        }
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}

// this adds a new column called dateFormattedString combining Year Month and DayofMonth
// will be used for the new holidayProximity feature
class formattedDate(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): formattedDate = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("formattedDate"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)
        
        var usableDF = df.toDF
        if ((columnExists(usableDF, "Year")) && (columnExists(usableDF, "Month")) && (columnExists(usableDF, "DayofMonth")))
            usableDF = usableDF.withColumn("dateFormattedString",
                concat(
                    col("DayofMonth").cast("String"),
                    lit("-"),
                    col("Month").cast("String"),
                    lit("-"),
                    col("Year").cast("String")
                )
            )
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}

// this tries to convert originally string variables (due to presence of NA) to ints
class stringTypesToInt(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): stringTypesToInt = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("stringTypesToInt"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)
        
        val shouldBeIntTypes = Array("ArrDelay", "TaxiOut", "DepDelay")
        var usableDF = df.toDF
        for ( columnName <- shouldBeIntTypes ){
            if (columnExists(usableDF, columnName))
            usableDF = usableDF.withColumn(columnName, col(columnName).cast("integer"))
        }
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}

// this converts hour dates from strings to ints by computing the according minutes
class stringHoursToInt(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): stringHoursToInt = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("stringHoursToInt"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    def convertStringTimeUDF() : org.apache.spark.sql.expressions.UserDefinedFunction = {
        return udf( (s : String) => {
            try{
                s.takeRight(2).toInt + s.reverse.slice(2,4).reverse.toInt * 60
            } 
            catch{ 
                case e: java.lang.NumberFormatException =>
                    0
            }
    })}

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        val possibleEmptyColumns = df.
            schema.fields.filter(
                x => x.dataType == org.apache.spark.sql.types.StringType
            ).map(_.name)

        
        val stringTimeTypes = Array("DepTime", "CRSDepTime", "CRSArrTime")
        var usableDF = df.toDF
        for ( columnName <- stringTimeTypes ){
            if (columnExists(usableDF, columnName))
                usableDF = usableDF.withColumn(columnName, convertStringTimeUDF()(col(columnName)))
        }
        usableDF 
    }

    override def transformSchema(schema: StructType): StructType = schema
}

// numerify all string values for Origin Dest and UniqueCarrier
class countriesAndCarriersAndFlightNumToNumbers(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): countriesAndCarriersAndFlightNumToNumbers = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("countriesAndCarriersAndFlightNumToNumbers"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    def convertWordsToNumbers(element : String, hash : Array[String]) : Int = {
        hash.indexOf(element)
    }

    def createConversionUDF(hash : Array[String]) : org.apache.spark.sql.expressions.UserDefinedFunction = {
        return udf{element : String => convertWordsToNumbers(element, hash)}
    }

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        if (columnExists(usableDF, "Origin") && (columnExists(usableDF, "Dest"))){
            val origKeys = usableDF.select(col("Origin")).distinct.collect.flatMap(_.toSeq).map(_.toString)
            val destKeys = usableDF.select(col("Dest")).distinct.collect.flatMap(_.toSeq).map(_.toString)
            val airportKeys = (origKeys ++ destKeys).distinct
            usableDF = usableDF.withColumn("Origin", createConversionUDF(airportKeys)(col("Origin")))
        } else if (columnExists(usableDF, "Dest")){
        val destKeys = usableDF.select(col("Dest")).distinct.collect.flatMap(_.toSeq).map(_.toString)
        usableDF = usableDF.withColumn("Dest", createConversionUDF(destKeys)(col("Dest")))
        } else if (columnExists(usableDF, "Origin")){
        val origKeys = usableDF.select(col("Origin")).distinct.collect.flatMap(_.toSeq).map(_.toString)
        usableDF = usableDF.withColumn("Origin", createConversionUDF(origKeys)(col("Origin")))
        }

        if (columnExists(usableDF, "UniqueCarrier")){
        val companiesKeys = usableDF.select(col("UniqueCarrier")).distinct.collect.flatMap(_.toSeq).map(_.toString)
        usableDF = usableDF.withColumn("UniqueCarrier", createConversionUDF(companiesKeys)(col("UniqueCarrier")))
        }

        if (columnExists(usableDF, "FlightNum")){
        val flightNumKeys = usableDF.select(col("FlightNum")).distinct.collect.flatMap(_.toSeq).map(_.toString)
        usableDF = usableDF.withColumn("FlightNum", createConversionUDF(flightNumKeys)(col("FlightNum")))
        }
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}

// talked about this above
class proximityToHoliday(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): proximityToHoliday = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("proximityToHoliday"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    def proximityToHoliday( element : String, holidays : Array[LocalDate]) : Int = {
        val format = DateTimeFormatter.ofPattern("d-M-y")
        val localDate = LocalDate.parse(element, format)
        var minimum = 999.toLong
        for ( holiday <- holidays ){
            val auxDiff = (DAYS.between(localDate, holiday))
            if (Math.abs(auxDiff) < minimum){
                minimum = auxDiff
            }
        }
        minimum.toInt
    }

    def proximityToHolidayUDF(holidays : Array[LocalDate]) : org.apache.spark.sql.expressions.UserDefinedFunction = {
        return udf( (s : String) => proximityToHoliday(s, holidays))
    }

    def computeHolidaysForYears(presentYears : Array[String]) : Array[LocalDate] = {
        var holidays = Array[LocalDate]()
        val holidayCalendarObj = new holidayCalendar
        for (year <- presentYears){
            val auxHolidayList = holidayCalendarObj.holidaysForYear(year)
            holidays = holidays ++ auxHolidayList
        }
        holidays
    }

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        if (columnExists(usableDF, "Year")){
            val presentYears = usableDF.select(col("Year")).distinct.collect.flatMap(_.toSeq).map(_.toString)
            if (presentYears.length == 1){
                usableDF = usableDF.drop("Year")
            }
            val holidays = computeHolidaysForYears(presentYears)
            // use the previously created column to create a proximityToHoliday column
            if (columnExists(usableDF, "dateFormattedString")){
                usableDF = usableDF.withColumn(
                    "proximityToHoliday", 
                    proximityToHolidayUDF(holidays)(col("dateFormattedString"))
                ).drop("dateFormattedString")
            }
        }
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}

// added an interaction variable between DepDelay and TaxiOut after multivariate analysis
class depDelayTaxiOutInteraction(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): depDelayTaxiOutInteraction = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("depDelayTaxiOutInteraction"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        if ((columnExists(usableDF, "DepTime")) && (columnExists(usableDF, "CRSDepTime")) && (columnExists(usableDF, "TaxiOut")))
            usableDF = usableDF.withColumn("depDelayTaxiOutInteraction", (col("DepTime") - col("CRSDepTime")) * col("TaxiOut"))
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
} 

// added an interaction variable between Origin and Dest after multivariate analysis
class origDestInteraction(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): origDestInteraction = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("origDestInteraction"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        if ((columnExists(usableDF, "Origin")) && (columnExists(usableDF, "Dest")))
            usableDF = usableDF.withColumn("origDestInteraction", col("Origin") * col("Dest"))
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}

class monthDayofMonthInteraction(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): monthDayofMonthInteraction = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("monthDayofMonthInteraction"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        if ((columnExists(usableDF, "Month")) && (columnExists(usableDF, "DayofMonth")))
            usableDF = usableDF.withColumn("monthDayofMonthInteraction", col("Month") * col("DayofMonth"))
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}

class depDelayOrigInteraction(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): depDelayOrigInteraction = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("depDelayOrigInteraction"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        if ((columnExists(usableDF, "DepTime")) && (columnExists(usableDF, "CRSDepTime")) && (columnExists(usableDF, "Origin")))
            usableDF = usableDF.withColumn("depDelayOrigInteraction", (col("DepTime") - col("CRSDepTime")) * col("Origin"))
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}


class depDelayCRSArrTimegInteraction(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): depDelayCRSArrTimegInteraction = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("depDelayCRSArrTimegInteraction"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        if ((columnExists(usableDF, "DepTime")) && (columnExists(usableDF, "CRSDepTime")) && (columnExists(usableDF, "Origin")))
            usableDF = usableDF.withColumn("depDelayCRSArrTimegInteraction", (col("DepTime") * col("CRSArrTime")))
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}

class dropRemainingString(override val uid: String) extends Transformer{    
    override def copy(extra: ParamMap): dropRemainingString = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("dropRemainingString"))

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        val possibleEmptyColumns = df.
            schema.fields.filter(
                x => x.dataType == org.apache.spark.sql.types.StringType
            ).map(_.name)
        for (columnName <- possibleEmptyColumns){
            usableDF = usableDF.drop(columnName)
        }
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}