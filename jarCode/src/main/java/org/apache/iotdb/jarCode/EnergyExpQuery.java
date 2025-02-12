package org.apache.iotdb.jarCode;

import java.io.IOException;
import java.io.PrintWriter;
import org.apache.iotdb.rpc.IoTDBConnectionException;
import org.apache.iotdb.rpc.StatementExecutionException;
import org.apache.iotdb.session.Session;
import org.apache.iotdb.session.SessionDataSet;
import org.apache.iotdb.tsfile.read.common.RowRecord;
import org.apache.thrift.TException;

public class EnergyExpQuery {

  // * (1) min_value(%1$s),max_value(%1$s),min_time(%1$s),max_time(%1$s),first_value(%1$s),last_value(%1$s)
  //       => Don't change the sequence of the above six aggregates!
  // * (2) NOTE the time unit of interval. Update for different datasets!
  private static final String SQL =
      "SELECT min_value(%1$s), max_value(%1$s),min_time(%1$s), max_time(%1$s), first_value(%1$s), "
          + "last_value(%1$s) FROM %2$s group by ([%3$d,%4$d),%5$d%6$s)";
  // note the time precision unit is also parameterized

  private static final String M4_UDF = "select M4(%1$s,'tqs'='%3$d','tqe'='%4$d','w'='%5$d') from %2$s where time>=%3$d and time<%4$d";

  private static final String MinMax_UDF = "select MinMax(%1$s,'tqs'='%3$d','tqe'='%4$d','w'='%5$d') from %2$s where time>=%3$d and time<%4$d";

  private static final String LTTB_UDF = "select Sample(%1$s,'method'='triangle','k'='%5$d') from %2$s where time>=%3$d and time<%4$d";


  public static Session session;

  // Usage: java -jar xx.jar
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
    // m数量
    int m = Integer.parseInt(args[6]);
    System.out.println("[QueryData] m=" + m);

    String approach = args[7]; // case sensitive

    long minTime;
    long maxTime;

    // fix minTime as dataMinTime, not random
    minTime = dataMinTime;
    if (approach.contains("UDF")) {
      long interval;
      if (range >= (dataMaxTime - dataMinTime)) {
        interval = (long) Math.ceil((double) (dataMaxTime - dataMinTime) / m);
      } else {
        interval = (long) Math.ceil((double) range / m);
      }
      maxTime = minTime + interval * m; // so that maxTime-minTime is integer multiply of m
    } else {
      if (range >= (dataMaxTime - dataMinTime)) {
        maxTime = dataMaxTime;
      } else {
        // randomize between [dataMinTime, dataMaxTime-range]
        maxTime = minTime + range;
      }
    }

    long tri_interval = (long) Math.floor((maxTime - minTime) * 1.0 / m);

    String sql;
    switch (approach) {
      case "MinMax":
        long minmax_interval = tri_interval * 2;
        sql = String.format(SQL, measurement, device, minTime, maxTime, minmax_interval,
            timestamp_precision);
        break;
      case "M4":
        long m4_interval = tri_interval * 4;
        sql = String.format(SQL, measurement, device, minTime, maxTime, m4_interval,
            timestamp_precision);
        break;
      case "LTTB":
      case "ILTS":
      case "SimPiece": // tri_interval is omitted by SimPiece, use epsilon control
      case "SC": // tri_interval is omitted by ShrinkingCone, use epsilon control
      case "FSW": // tri_interval is omitted by FSW, use epsilon control
      case "Uniform": // same interval
      case "Visval": // transform into m automatically
        sql = String.format(SQL, measurement, device, minTime, maxTime, tri_interval,
            timestamp_precision);
        break;
      case "MinMaxLTTB":
        int rps = 2;
        long minmax_preselect_interval = tri_interval / (rps / 2);
        sql = String.format(SQL, measurement, device, minTime, maxTime, minmax_preselect_interval,
            timestamp_precision);
        break;
      case "MinMax_UDF":
        sql = String.format(MinMax_UDF, measurement, device, minTime, maxTime, m / 2);
        break;
      case "M4_UDF":
        sql = String.format(M4_UDF, measurement, device, minTime, maxTime, m / 4);
        break;
      case "LTTB_UDF":
        sql = String.format(LTTB_UDF, measurement, device, minTime, maxTime, m); // note 4w
        break;
      case "MinMaxLTTB_UDF":
      default:
        throw new IOException("Approach wrong.");
    }

    System.out.println("[QueryData] approach=" + approach);
    if (!approach.contains("UDF")) {
      System.out.printf(
          "MAKE SURE you have set the enable_tri as %s in `iotdb-engine.properties`!%n",
          approach);
    }

    boolean save_query_result = Boolean.parseBoolean(args[8]);
    System.out.println("[QueryData] save_query_result=" + save_query_result);

    String save_query_path = args[9];
    System.out.println("[QueryData] save_query_path=" + save_query_path);
    if (!save_query_result) {
      throw new IOException("Should save query result!");
    }

    session = new Session("127.0.0.1", 6667, "root", "root");
    session.open(false);

    // Set it big to avoid multiple fetch, which is very important.
    // Because the IOMonitor implemented in IoTDB does not cover the fetchResults operator yet.
    // As M4 already does data reduction, so even the w is very big such as 8000, the returned
    // query result size is no more than 8000*4=32000.
    session.setFetchSize(1000000);

    PrintWriter printWriter = new PrintWriter(save_query_path);
    long startTime = System.nanoTime();

    SessionDataSet dataSet = session.executeQueryStatement(sql);
    while (dataSet.hasNext()) { // this way avoid constructing rowRecord
      RowRecord rowRecord = dataSet.next();
      String input = rowRecord.getFields().get(0).getStringValue();
      // e.g., 0.0[0],0.004[7000],0.0[10000],
      if (input.endsWith(",")) {
        input = input.substring(0, input.length() - 1);
      }
      String[] pairs = input.split(",");
      for (String pair : pairs) {
        String[] parts = pair.split("\\[|\\]"); // 使用正则表达式分割字符串
        String v = parts[0].trim();
        String t = parts[1].trim();
        printWriter.println(t + "," + v);
      }
    }
    long elapsedTimeNanoSec = System.nanoTime() - startTime;
    System.out.println("[1-ns]ClientElapsedTime," + elapsedTimeNanoSec);

    dataSet.closeOperationHandle();
    session.close();
    printWriter.close();
  }
}
