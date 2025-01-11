package org.apache.iotdb.jarCode;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import org.apache.iotdb.rpc.IoTDBConnectionException;
import org.apache.iotdb.rpc.StatementExecutionException;
import org.apache.iotdb.session.Session;
import org.apache.iotdb.session.SessionDataSet;
import org.apache.iotdb.session.SessionDataSet.DataIterator;
import org.apache.iotdb.tsfile.read.common.RowRecord;
import org.apache.thrift.TException;

public class QueryData {

  // * (1) min_value(%1$s),max_value(%1$s),min_time(%1$s),max_time(%1$s),first_value(%1$s),last_value(%1$s)
  //       => Don't change the sequence of the above six aggregates!
  // * (2) NOTE the time unit of interval. Update for different datasets!
  private static final String SQL =
      "SELECT min_value(%1$s), max_value(%1$s),min_time(%1$s), max_time(%1$s), first_value(%1$s), "
          + "last_value(%1$s) FROM %2$s group by ([%3$d,%4$d),%5$d%6$s)";
  // note the time precision unit is also parameterized

//  private static final String M4_UDF = "select M4(%1$s,'tqs'='%3$d','tqe'='%4$d','w'='%5$d') from %2$s where time>=%3$d and time<%4$d";

//  private static final String MinMax_UDF = "select MinMax(%1$s,'tqs'='%3$d','tqe'='%4$d','w'='%5$d') from %2$s where time>=%3$d and time<%4$d";

//  private static final String LTTB_UDF = "select Sample(%1$s,'method'='triangle','k'='%5$d') from %2$s where time>=%3$d and time<%4$d";


  public static Session session;

  // Usage: java -jar QueryData-0.12.4.jar
  // device measurement timestamp_precision dataMinTime dataMaxTime range m approach save_query_result save_query_path
  public static void main(String[] args)
      throws IoTDBConnectionException, StatementExecutionException, TException, IOException {
    String device = args[0];
    System.out.println("[QueryData] device=" + device);
    String measurement = args[1];
    System.out.println("[QueryData] measurement=" + measurement);

    String timestamp_precision = args[2]; // ns, us, ms
    timestamp_precision = timestamp_precision.toLowerCase();
    System.out.println("[QueryData] timestamp_precision=" + timestamp_precision);
    if (!timestamp_precision.toLowerCase().equals("ns") && !timestamp_precision.toLowerCase()
        .equals("us") && !timestamp_precision.toLowerCase().equals("ms")) {
      throw new IOException("timestamp_precision only accepts ns,us,ms.");
    }

    // used to bound tqs random position
    long dataMinTime = Long.parseLong(args[3]);
    System.out.println("[QueryData] dataMinTime=" + dataMinTime);
    long dataMaxTime = Long.parseLong(args[4]);
    System.out.println("[QueryData] dataMaxTime=" + dataMaxTime);

    // [tqs,tqe) range length, i.e., tqe-tqs
    long range = Long.parseLong(args[5]);
    System.out.println("[QueryData] query range=" + range);

    int m = Integer.parseInt(args[6]);
    System.out.println("[QueryData] m=" + m);

    String approach = args[7]; // case sensitive
    System.out.println("[QueryData] approach=" + approach);
    System.out.printf(
        "MAKE SURE you have set the enable_tri as %s in `iotdb-engine.properties`!%n",
        approach);

    boolean save_query_result = Boolean.parseBoolean(args[8]);
    System.out.println("[QueryData] save_query_result=" + save_query_result);

    String save_query_path = args[9];
    System.out.println("[QueryData] save_query_path=" + save_query_path);

    String om3_query_dir = "";
    if (args.length > 10) {
      om3_query_dir = args[10];
    }
    System.out.println("[QueryData] om3_query_dir=" + om3_query_dir);

    long minTime;
    long maxTime;

    // fix minTime as dataMinTime, not random
    minTime = dataMinTime;
    if (range >= (dataMaxTime - dataMinTime)) {
      maxTime = dataMaxTime;
    } else {
      maxTime = minTime + range;
    }

    long tri_interval = (long) Math.floor((maxTime - minTime) * 1.0 / m);

    List<String> sql = new ArrayList<>();
    switch (approach) {
      case "MinMax":
        long minmax_interval = tri_interval * 2;
        sql.add(String.format(SQL, measurement, device, minTime, maxTime, minmax_interval,
            timestamp_precision));
        break;
      case "M4":
        long m4_interval = tri_interval * 4;
        sql.add(String.format(SQL, measurement, device, minTime, maxTime, m4_interval,
            timestamp_precision));
        break;
      case "LTTB":
      case "ILTS":
      case "SimPiece": // tri_interval is omitted by SimPiece, use epsilon control
      case "SC": // tri_interval is omitted by ShrinkingCone, use epsilon control
      case "FSW": // tri_interval is omitted by FSW, use epsilon control
      case "Uniform": // same interval
      case "Visval": // transform into m automatically
        sql.add(String.format(SQL, measurement, device, minTime, maxTime, tri_interval,
            timestamp_precision));
        break;
      case "MinMaxLTTB":
        int rps = 2;
        long minmax_preselect_interval = tri_interval / (rps / 2);
        sql.add(String.format(SQL, measurement, device, minTime, maxTime, minmax_preselect_interval,
            timestamp_precision));
        break;
      case "OM3":
        String sqlQueryTemplate = "SELECT %s FROM %s WHERE timestamp in (%s);";
        try (BufferedReader br = new BufferedReader(new FileReader(om3_query_dir))) {
          List<String> numbers = new ArrayList<>();
          String line;
          while ((line = br.readLine()) != null) {
            String number = line.split(",")[0]; // 假设第一列没有逗号
            numbers.add(number);
          }
          String numbersStr = String.join(",", numbers);
          // measurement = "root.QlossMin.targetDevice.test, root.QlossMax.targetDevice.test"
          String measure1 = measurement.split(",")[0].replace("root.", "");
          String measure2 = measurement.split(",")[1].replace("root.", "");
          sql.add(String.format(sqlQueryTemplate, measure1, device, numbersStr));
          sql.add(String.format(sqlQueryTemplate, measure2, device, numbersStr));

        } catch (IOException e) {
          throw new IOException(e);
        }
        break;
      default:
        throw new IOException("Not supported approach.");
    }

    session = new Session("127.0.0.1", 6667, "root", "root");
    session.open(false);

    // Set it big to avoid multiple fetch, which is very important.
    // Because the IOMonitor implemented in IoTDB does not cover the fetchResults operator yet.
    // As M4 already does data reduction, so even the w is very big such as 8000, the returned
    // query result size is no more than 8000*4=32000.
    session.setFetchSize(1000000);

    if (!save_query_result) {
      long c = 0;
      long startTime = System.nanoTime();
      for (String query : sql) {
        SessionDataSet dataSet = session.executeQueryStatement(query);
        DataIterator ite = dataSet.iterator();
        while (ite.next()) { // this way avoid constructing rowRecord
          c++;
        }
      }
      long elapsedTimeNanoSec = System.nanoTime() - startTime;
      System.out.println("[1-ns]ClientElapsedTime," + elapsedTimeNanoSec);

      SessionDataSet dataSet = session.executeFinish();
      String info = dataSet.getFinishResult();
      // don't add more string to this output, as ProcessResult code depends on this.
      System.out.println(info);
      System.out.println("[QueryData] query result line number=" + c);

      dataSet.closeOperationHandle();
      session.close();
    } else {
      PrintWriter printWriter = new PrintWriter(save_query_path);
      long c = 0;
      long startTime = System.nanoTime();
      for (String query : sql) {
        SessionDataSet dataSet = session.executeQueryStatement(query);
        while (dataSet.hasNext()) { // this way avoid constructing rowRecord
          RowRecord rowRecord = dataSet.next();
          printWriter.println(rowRecord.getFields().get(0).getStringValue());
          c++;
        }
      }
      long elapsedTimeNanoSec = System.nanoTime() - startTime;
      System.out.println("[1-ns]ClientElapsedTime," + elapsedTimeNanoSec);

      SessionDataSet dataSet = session.executeFinish();
      String info = dataSet.getFinishResult();
      // don't add more string to this output, as ProcessResult code depends on this.
      System.out.println(info);
      System.out.println("[QueryData] query result line number=" + c);

      dataSet.closeOperationHandle();
      session.close();
      printWriter.close();
    }
  }
}
