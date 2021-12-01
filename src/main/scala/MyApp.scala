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
import org.apache.spark.ml.linalg.{Vector, Vectors}

object MyApp extends remainingMethodsClass{

    def main(args : Array[String]) {
        Logger.getLogger("org").setLevel(Level.ERROR)
        val conf = new SparkConf().setAppName("Deliverable")
        val sc = new SparkContext(conf)
            .setCheckpointDir("/tmp/checkpoints")
        val spark = SparkSession.builder()
            .appName("Big Data Proj")
            .getOrCreate()

        // read the input file
        val auxOutput = readInput(args, spark)
        // the first object of the tuple is the dataframe
        var rawDF = auxOutput._1
        // the 2nd object of the tuple is the missing/incorrect columns
        val wrongColumnsList = auxOutput._2
        // the 3rd object of the tuple is the chosen machine laerning alg
        val machineLearningAlg = auxOutput._3
        println(machineLearningAlg)
        val extraDatasetPath = auxOutput._4
        // throw error if not all columns are present or if file is empty
        checkIfEmpty(rawDF)        
        // we try to remove duplicate rows so no information leaks from test to training
        rawDF = rawDF.distinct

        // if there was an extra file given, try to use that information
        if (extraDatasetPath != ""){
            rawDF = readExtraFile(rawDF, extraDatasetPath, spark)
            if ((columnExists(rawDF, "model")) && (columnExists(rawDF, "engine_type"))){
                rawDF = rawDF.na.fill(0, Seq("model"))
                rawDF = rawDF.na.fill(0, Seq("engine_type"))
            }
        }
        
        val dropperTransformer = new dropper().setHandleInvalid("skip") // drop all forbidden/useless columns
        dropperTransformer.wrongColumnsList = wrongColumnsList // adding the columns that really should be removed
        val emptyColumnsAndFieldsTransformer = new emptyColumnsAndFields().setHandleInvalid("skip") // get all string columns && remove all remaining null fields OR drop column with too many NAs
        val formattedDateTransformer = new formattedDate().setHandleInvalid("skip")  // add column for further analysis (proximityToHolidayTransformer   )
        val stringTypesToIntTransformer = new stringTypesToInt().setHandleInvalid("skip") // cast possible string columns to integer
        val stringHoursToIntTransformer = new stringHoursToInt().setHandleInvalid("skip") // cast the CRSDepTime to int using magic formula
        val modelEngineTypeTransformer = new modelEngineType().setHandleInvalid("skip") // change strings to numbers for discrete values for those 2 columns
        val countriesAndCarriersAndFlightNumToNumbersTransformer = new countriesAndCarriersAndFlightNumToNumbers().setHandleInvalid("skip") // here we collect all the countries/carrierCodes present in origin/dest/codes & change strings to numbers
        val proximityToHolidayTransformer = new proximityToHoliday().setHandleInvalid("skip") // here we compute the US holiday/event list for those Year values (holidayCalendar class defined below)
        var pipelineStages : Array[Transformer] = Array(
            dropperTransformer,
            emptyColumnsAndFieldsTransformer,
            formattedDateTransformer,
            stringTypesToIntTransformer,
            stringHoursToIntTransformer,
            modelEngineTypeTransformer,
            countriesAndCarriersAndFlightNumToNumbersTransformer,
            proximityToHolidayTransformer
        )

        // if the machine learning alg is lr then we also add interaction variables 
        if (machineLearningAlg.toLowerCase == "lr"){
            val depDelayTaxiOutInteractionTransformer = new depDelayTaxiOutInteraction().setHandleInvalid("skip") // create intraction variable between old DepDelay and TaxiOut
            pipelineStages = pipelineStages :+ depDelayTaxiOutInteractionTransformer
            val origDestInteractionTransformer = new origDestInteraction().setHandleInvalid("skip") // create intraction variable between Origin and Dest
            pipelineStages = pipelineStages :+ origDestInteractionTransformer
            val monthDayofMonthInteractionTransformer = new  monthDayofMonthInteraction().setHandleInvalid("skip") // create interaction variable between Month and DayofMonth
            pipelineStages = pipelineStages :+ monthDayofMonthInteractionTransformer
            val depDelayOrigInteractionTransformer = new depDelayOrigInteraction().setHandleInvalid("skip")
            pipelineStages = pipelineStages :+ depDelayOrigInteractionTransformer
            val depDelayCRSDepTimeInteractionTransformer = new depDelayCRSDepTimeInteraction().setHandleInvalid("skip")
            pipelineStages = pipelineStages :+ depDelayCRSDepTimeInteractionTransformer
        }
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
        var scaledTraining = scalerModel.transform(training)
        var scaledTest = scalerModel.transform(test)

        if (machineLearningAlg == "lr"){
            val addSquaredTermsTransformer = new addSquaredTerms().setHandleInvalid("skip")
            val polynomialPipeline = new Pipeline().setStages(Array(addSquaredTermsTransformer))
            
            val polynomialPipelineTraining = polynomialPipeline.fit(scaledTraining)
            scaledTraining = polynomialPipelineTraining.transform(scaledTraining)
            val polynomialPipelineTest = polynomialPipeline.fit(scaledTest)
            scaledTest = polynomialPipelineTest.transform(scaledTest)
        }

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

    def readInput(args: Array[String], spark: SparkSession) : (DataFrame, Array[String], String, String) = {
        // RULE: everything that is not CRScheduled is String.
        if (args.length >= 1) {
            if (args(0).takeRight(4) != ".csv") {
                throw new Exception("File format is not supported. Please use .csv")
            }
            // first we only read the headers to see what the order of the columns is
            // we also check for wrong names or just missing columns
            val sourceFile = Source.fromFile(args(0)) 
            val headerLine = sourceFile.getLines.take(1).toArray
            sourceFile.close
            val headersRaw = headerLine(0).split(",")
            val headersFinal = headersRaw.map(_.replaceAll("\\W", ""))
            // this is the hash containing what the names should be and their associated type
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
            // the schemaArray object will be used to impose a schema array when reading the actual data
            var schemaArray : Array[StructField] = Array() 
            // the wrongColumnArray object will be used to know what columns to drop after having imposed the schema
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
            var extraDatasetPath = ""
            if (args.length >= 2){
                machineLearningAlg = args(1)
                if (args.length >= 3){
                    extraDatasetPath = args(2)
                }
            }
            return (spark.read.schema(genericSchema).options(Map("header"->"true")).csv(args(0)), wrongColumnArray, machineLearningAlg, extraDatasetPath)
        } 
        else {
            throw new Exception("locaiton of the input file is missing or incorrect")
        }
    }

    // reads extra file if necessary
    def readExtraFile( df : DataFrame, input_path : String, spark: SparkSession) : DataFrame = {
        var usableDF = df
        try{
            if (input_path.takeRight(4) != ".csv") {
                throw new Exception("File format is not supported. Please use .csv")
            }
            val sourceFile = Source.fromFile(input_path) 
            val headerLine = sourceFile.getLines.take(1).toArray
            sourceFile.close
            val headersRaw = headerLine(0).split(",")
            val headersFinal = headersRaw.map(_.replaceAll("\\W", ""))

            val correctHeaders = Array("tailnum", "type", "manufacturer", "issue_date", "model", "status", "aircraft_type", "engine_type", "year")
            var index = 0
            for (currentHeaders <- correctHeaders){
                if (currentHeaders != headersFinal(index))
                    throw new Exception("Something wrong with the column names. It's not like the original, so the new columns won't be used.")
                index += 1
            }
            val secondarySchema = Array(
                StructField("tailnum", StringType, true),
                StructField("type", StringType, true),
                StructField("manufacturer", StringType, true),
                StructField("issue_date", StringType, true),
                StructField("model", StringType, true),
                StructField("status", StringType, true),
                StructField("aircraft_type", StringType, true),
                StructField("engine_type", StringType, true),
                StructField("year", StringType, true)
            )
            val extraDataset = spark.read.schema(StructType(secondarySchema)).options(Map("header"->"true")).csv(input_path)
            val onlyPlaneInfoDF = extraDataset.select(
                    col("tailnum"), 
                    col("model"), 
                    col("engine_type")
                ).filter((!col("model").isNull) && (!col("engine_type").isNull))
            onlyPlaneInfoDF.first
            // joins the original dataframe with the new one in case the file is good.
            usableDF = usableDF.join(onlyPlaneInfoDF, usableDF("TailNum") === onlyPlaneInfoDF("tailnum")).drop("tailnum")
        } catch {
            case e: Exception =>
                    println("The extra dataset was not exactly the same as the one online. Therefore extra variables won't be added.")
        }
        return usableDF
    }

    // checks to make sure the input is good :) 
    def checkIfEmpty( df : DataFrame ) : Boolean = {
        try {
            // if this call doesn't work means the file is empty so it's useless
            df.first
        }catch{ 
            case e: java.util.NoSuchElementException =>
                throw new Exception("The file is, sadly, empty. Please choose a file containing actual data.")
        }
        return true
    }

    // this function has been created to centralize the creation of the CrossValidator
    // depending on the chosen learning algorithm
    def createMLModel( chosenAlgorithm : String ) : CrossValidator = {
        if (chosenAlgorithm.toLowerCase == "lr") {
            val algorithm = new LinearRegression().setFeaturesCol("scaledFeatures").setLabelCol("ArrDelay").setMaxIter(1000)
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
            val algorithm = new RandomForestRegressor().setLabelCol("ArrDelay").setFeaturesCol("scaledFeatures").setFeatureSubsetStrategy("auto").setImpurity("variance").setMaxBins(100)
            val paramGrid = new ParamGridBuilder()
            .addGrid(
                algorithm.maxDepth, 
                Array(
                    7, 9, 11
                )).addGrid(
                algorithm.numTrees,
                Array(
                    7, 9, 11
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
            "TailNum",    // also drop the TailNum column given that it's an ID and it's not useful forE
            "CRSArrTime", "DepTime", // drop because they're all in different timezones + we have CRSElapsedTime and DepDelay
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

            // if there are too many nulls/NA then the column is dropped in favor or retaining as many rows as possible - 30%
            // if there are less than 30% missing values, we just drop those rows.
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
        // converts string columns to integers
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

    // udf that converts the hour into an integer using a simple formula
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
        
        val stringTimeTypes = Array("CRSDepTime")
        var usableDF = df.toDF
        // just convert hour string columns to integers
        // the for is there in case we want to add extra hour string columns to be casted :D
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

    // function to be called inside udf
    def convertWordsToNumbers(element : String, hash : Array[String]) : Int = {
        hash.indexOf(element)
    }

    // function that returns a udf so we can call function with input params
    def createConversionUDF(hash : Array[String]) : org.apache.spark.sql.expressions.UserDefinedFunction = {
        return udf{element : String => convertWordsToNumbers(element, hash)}
    }

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        // check if origin/dest both exist and then change strings to ints
        if (columnExists(usableDF, "Origin") && (columnExists(usableDF, "Dest"))){
            // get origin present airports
            val origKeys = usableDF.select(col("Origin")).distinct.collect.flatMap(_.toSeq).map(_.toString)
            // get dest present airports
            val destKeys = usableDF.select(col("Dest")).distinct.collect.flatMap(_.toSeq).map(_.toString)
            // do a union between them because we want the same point of reference
            val airportKeys = (origKeys ++ destKeys).distinct
            // change the strings to ints using a (dictionary) map
            usableDF = usableDF.withColumn("Origin", createConversionUDF(airportKeys)(col("Origin")))
            usableDF = usableDF.withColumn("Dest", createConversionUDF(airportKeys)(col("Dest")))
        } else if (columnExists(usableDF, "Dest")){
        val destKeys = usableDF.select(col("Dest")).distinct.collect.flatMap(_.toSeq).map(_.toString)
        // change the strings to ints using a (dictionary) map
        usableDF = usableDF.withColumn("Dest", createConversionUDF(destKeys)(col("Dest")))
        } else if (columnExists(usableDF, "Origin")){
        val origKeys = usableDF.select(col("Origin")).distinct.collect.flatMap(_.toSeq).map(_.toString)
        // change the strings to ints using a (dictionary) map
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

    // function to be called in the UDF
    def proximityToHoliday( element : String, holidays : Array[LocalDate]) : Int = {
        val format = DateTimeFormatter.ofPattern("d-M-y")
        val localDate = LocalDate.parse(element, format)
        var minimum = 999.toLong
        // find the closest holiday and find the day difference between the 2 dates and store it
        for ( holiday <- holidays ){
            val auxDiff = (DAYS.between(localDate, holiday))
            if (Math.abs(auxDiff) < minimum){
                minimum = auxDiff
            }
        }
        minimum.toInt
    }

    // function that returns a udf so that we can also have input params
    def proximityToHolidayUDF(holidays : Array[LocalDate]) : org.apache.spark.sql.expressions.UserDefinedFunction = {
        return udf( (s : String) => proximityToHoliday(s, holidays))
    }

    // function that adds holidays to an array
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
            // get every present year
            val presentYears = usableDF.select(col("Year")).distinct.collect.flatMap(_.toSeq).map(_.toString)
            // drop year if there s only 1 year present
            if (presentYears.length == 1){
                usableDF = usableDF.drop("Year")
            }
            // get holiday dates for that year
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
        // self-explanatory, we add a new column that is the multiplication between depdelay and taxiout
        if ((columnExists(usableDF, "DepDelay")) && (columnExists(usableDF, "TaxiOut")))
            usableDF = usableDF.withColumn("depDelayTaxiOutInteraction", (col("DepDelay") * col("TaxiOut")))
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
} 

// added an interaction variable between Origin and Dest
class origDestInteraction(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): origDestInteraction = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("origDestInteraction"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        // self-explanatory, we add a new column that is the multiplication between origin and dest
        if ((columnExists(usableDF, "Origin")) && (columnExists(usableDF, "Dest")))
            usableDF = usableDF.withColumn("origDestInteraction", col("Origin") * col("Dest"))
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}

// added an interaction variable between Month and DayofMonth
class monthDayofMonthInteraction(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): monthDayofMonthInteraction = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("monthDayofMonthInteraction"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        // self-explanatory, we add a new column that is the multiplication between month and dayofmonth
        if ((columnExists(usableDF, "Month")) && (columnExists(usableDF, "DayofMonth")))
            usableDF = usableDF.withColumn("monthDayofMonthInteraction", col("Month") * col("DayofMonth"))
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}

// added an interaction variable between DepDelay and Origin
class depDelayOrigInteraction(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): depDelayOrigInteraction = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("depDelayOrigInteraction"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        // self-explanatory, we add a new column that is the multiplication between depdelay and origin
        if ((columnExists(usableDF, "DepDelay")) && (columnExists(usableDF, "Origin")))
            usableDF = usableDF.withColumn("depDelayOrigInteraction", (col("DepDelay") * col("Origin")))
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}

// added an interaction variable between DepDelay and CRSDepTime
class depDelayCRSDepTimeInteraction(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): depDelayCRSDepTimeInteraction = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("depDelayCRSDepTimeInteraction"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        // self-explanatory, we add a new column that is the multiplication between depdelay and crsdeptime
        if ((columnExists(usableDF, "DepDelay")) && (columnExists(usableDF, "CRSDepTime")))
            usableDF = usableDF.withColumn("depDelayCRSDepTimeInteraction", (col("DepDelay") * col("CRSDepTime")))
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}

// just drop everything that is still string (for whatever reason) by the end of the pipeline
class dropRemainingString(override val uid: String) extends Transformer{    
    override def copy(extra: ParamMap): dropRemainingString = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("dropRemainingString"))

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        // get all column that are still strings 
        val possibleStringColumns = df.
            schema.fields.filter(
                x => x.dataType == org.apache.spark.sql.types.StringType
            ).map(_.name)
        // drop em
        for (columnName <- possibleStringColumns){
            usableDF = usableDF.drop(columnName)
        }
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}

// this transformer transforms the model/ engine type from strings to ints if they exist
class modelEngineType(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): modelEngineType = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("modelEngineType"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    // create function to be called in the UDF function
    def convertWordsToNumbers(element : String, hash : Array[String]) : Int = {
        hash.indexOf(element)
    }

    // function that returns a udf so that we can also have input params
    def createConversionUDF(hash : Array[String]) : org.apache.spark.sql.expressions.UserDefinedFunction = {
        return udf{element : String => convertWordsToNumbers(element, hash)}
    }

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)


        var usableDF = df.toDF
        if (columnExists(usableDF, "model")){
        // get a map containing all keys mapped to their location and use that to cast strings to integer
        val modelKeys = usableDF.select(col("model")).distinct.collect.flatMap(_.toSeq).map(_.toString)
        usableDF = usableDF.withColumn("model", createConversionUDF(modelKeys)(col("model")))
        }

        if (columnExists(usableDF, "engine_type")){
        // get a map containing all keys mapped to their location and use that to cast strings to integer            
        val engineKeys = usableDF.select(col("engine_type")).distinct.collect.flatMap(_.toSeq).map(_.toString)
        usableDF = usableDF.withColumn("engine_type", createConversionUDF(engineKeys)(col("engine_type")))
        }
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}

// class name is self-explanatory
class addSecondDegreeTerms(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): addSecondDegreeTerms = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("addSecondDegreeTerms"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        var variableNames = usableDF.columns.filter(_ != "ArrDelay")
        // if the machine learning alg is lr then we add 2nd degree terms to also model nonlinearity
        for (colName <- variableNames){
            if (!(colName.toLowerCase contains "interaction")){
                var new_name = colName + "Squared"
                usableDF = usableDF.withColumn(new_name, col(colName) * col(colName))
            }
        }
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}

// added a transformer to add square terms in case of LR
class addSquaredTerms(override val uid: String) extends Transformer with columnExisting with HasHandleInvalid{    
    override def copy(extra: ParamMap): addSquaredTerms = defaultCopy(extra)

    def this() = this(Identifiable.randomUID("addSquaredTerms"))

    def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

    // function to be called in the UDF
    def createNewVector(x : Vector, columnNames : scala.collection.immutable.Map[Int, String]) : Vector = {
        var index = 0 
        var auxArray = x.toArray // get the original array
        val arrayForIteration = x.toArray // get the same one for iterating over
        // iterate over the array, if the variable is not an interaction one, add a square term to the new array
        for (item <- arrayForIteration){
            try{
                if (!(columnNames(index).toLowerCase contains ("interaction"))){
                    auxArray = auxArray :+ item * item
                }
            } catch {case e: Exception => index = index}
            index += 1
        }
        // create the new vector from the new array
        val newVector  = Vectors.dense(auxArray)
        // return it 
        newVector
    }
    
    // function that returns a udf so that we can also have input params
    def createSquaredTermsUDF(columnNames : scala.collection.immutable.Map[Int, String]) : org.apache.spark.sql.expressions.UserDefinedFunction = {
        return udf{ s : Vector => createNewVector(s, columnNames)}
    }

    override def transform(df: Dataset[_]) : DataFrame = {
        transformSchema(df.schema, logging = true)

        var usableDF = df.toDF
        if (columnExists(usableDF, "scaledFeatures")){  
            // get the columnNames so with their order in the features array so we can identify them in the UDF function
            val columnNames = usableDF.columns.filter( x => ((x != "features") && (x != "scaledFeatures") && (x != "proximityToHoliday"))).zipWithIndex.map( x => (x._2, x._1)).toMap
            usableDF = usableDF.withColumn("scaledFeatures", createSquaredTermsUDF(columnNames)(col("scaledFeatures")))
        }
        usableDF
    }

    override def transformSchema(schema: StructType): StructType = schema
}