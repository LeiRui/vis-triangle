package org.apache.iotdb.jarCode;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.iotdb.rpc.IoTDBConnectionException;
import org.apache.iotdb.rpc.StatementExecutionException;
import org.apache.iotdb.session.Session;
import org.apache.iotdb.tsfile.file.metadata.enums.TSDataType;
import org.apache.iotdb.tsfile.file.metadata.enums.TSEncoding;
import org.apache.iotdb.tsfile.write.record.Tablet;
import org.apache.iotdb.tsfile.write.schema.MeasurementSchema;

public class WriteDataOM3 {

  /**
   * Before writing data, make sure check the server parameter configurations.
   */
  // Usage: java -jar WriteDataMore.jar device measurement timestamp_precision dataType valueEncoding iotdb_chunk_point_size filePath
  // NEW: write multiple series as different devices
  public static void main(String[] args)
      throws IoTDBConnectionException, StatementExecutionException, IOException {
    String device = args[0];
    System.out.println("[WriteData] device=" + device);
    // device default to be three levels such as "root.HouseTwenty.targetDevice"
    // the first two levels form storage group, while the last level is device
    int dotCount = 0;
    for (int i = 0; i < device.length(); i++) {
      if (device.charAt(i) == '.') {
        dotCount++;
      }
    }
    if (dotCount != 2) { // not three levels
      throw new IOException("wrong device!");
    }

    String measurement = args[1];
    System.out.println("[WriteData] measurement=" + measurement);

    String timestamp_precision = args[2]; // ns, us, ms
    System.out.println("[WriteData] timestamp_precision=" + timestamp_precision);
    if (!timestamp_precision.toLowerCase().equals("ns") && !timestamp_precision.toLowerCase()
        .equals("us") && !timestamp_precision.toLowerCase().equals("ms")) {
      throw new IOException("timestamp_precision only accepts ns,us,ms.");
    }

    String dataType = args[3]; // double only
    System.out.println("[WriteData] dataType=" + dataType);
    TSDataType tsDataType;
    if (dataType.toLowerCase().equals("double")) {
      tsDataType = TSDataType.DOUBLE;
    } else {
      throw new IOException("Data type only accepts double right now.");
    }

    // value encoder
    String valueEncoding = args[4]; // RLE, GORILLA, PLAIN
    System.out.println("[WriteData] valueEncoding=" + valueEncoding);

    int iotdb_chunk_point_size = Integer.parseInt(args[5]);
    System.out.println("[WriteData] iotdb_chunk_point_size=" + iotdb_chunk_point_size);

    String filePath = args[6];
    System.out.println("[WriteData] filePath=" + filePath);

    int cntOther = 20;
    if (args.length >= 8) {
      cntOther = Integer.parseInt(args[7]);
    }
    System.out.println("[WriteData] cntOther=" + cntOther);

    int maxPointWritten = -1;
    if (args.length >= 9) {
      maxPointWritten = Integer.parseInt(args[8]);
    }
    System.out.println("[WriteData] maxPointWritten=" + maxPointWritten);

    int valueIdx = Integer.parseInt(args[9]);

    String otherMeasurement = "otherSensor";

    Session session = new Session("127.0.0.1", 6667, "root", "root");
    session.open(false);

    //"CREATE TIMESERIES root.vehicle.d0.s0 WITH DATATYPE=INT32, ENCODING=RLE"
    String createSql = String.format("CREATE TIMESERIES %s.%s WITH DATATYPE=%s, ENCODING=%s",
        device,
        measurement,
        tsDataType,
        valueEncoding
    );
    session.executeNonQueryStatement(createSql);

    // device default to be three levels such as "root.HouseTwenty.targetDevice"
    String storageGroup = device.substring(0, device.lastIndexOf('.'));
    String otherDevice = storageGroup + ".otherDevice"; // same storage group but different devices

    for (int i = 1; i <= cntOther; i++) { // note sensor name start from 1
      String createOtherSql = String.format(
          "CREATE TIMESERIES %s%d.%s WITH DATATYPE=%s, ENCODING=%s",
          otherDevice, i,   // same storage group but different devices
          otherMeasurement,  // sensor name
          tsDataType,
          valueEncoding
      );
      session.executeNonQueryStatement(createOtherSql);
    }

    // this is to make all following inserts unseq chunks
    if (timestamp_precision.toLowerCase().equals("ns")) {
      session.insertRecord(
          device,
          1683616109697000000L, // ns
          // NOTE UPDATE TIME DATATYPE! [[update]]. DONT USE System.nanoTime()!
          Collections.singletonList(measurement),
          Collections.singletonList(tsDataType), // NOTE UPDATE VALUE DATATYPE!
          parseValue("0", tsDataType)); // NOTE UPDATE VALUE DATATYPE!
      if (cntOther > 0) {
        for (int i = 1; i <= cntOther; i++) {
          session.insertRecord(
              otherDevice + i,
              1683616109697000000L, // ns
              // NOTE UPDATE TIME DATATYPE! [[update]]. DONT USE System.nanoTime()!
              Collections.singletonList(otherMeasurement),
              Collections.singletonList(tsDataType), // NOTE UPDATE VALUE DATATYPE!
              parseValue("0", tsDataType)); // NOTE UPDATE VALUE DATATYPE!
        }
      }
    } else if (timestamp_precision.toLowerCase().equals("us")) {
      session.insertRecord(
          device,
          1683616109697000L, // us
          // NOTE UPDATE TIME DATATYPE! [[update]]. DONT USE System.nanoTime()!
          Collections.singletonList(measurement),
          Collections.singletonList(tsDataType), // NOTE UPDATE VALUE DATATYPE!
          parseValue("0", tsDataType)); // NOTE UPDATE VALUE DATATYPE!
      if (cntOther > 0) {
        for (int i = 1; i <= cntOther; i++) {
          session.insertRecord(
              otherDevice + i,
              1683616109697000L, // us
              // NOTE UPDATE TIME DATATYPE! [[update]]. DONT USE System.nanoTime()!
              Collections.singletonList(otherMeasurement),
              Collections.singletonList(tsDataType), // NOTE UPDATE VALUE DATATYPE!
              parseValue("0", tsDataType)); // NOTE UPDATE VALUE DATATYPE!
        }
      }
    } else { // ms
      session.insertRecord(
          device,
          1683616109697L, // ms
          // NOTE UPDATE TIME DATATYPE! [[update]]. DONT USE System.nanoTime()!
          Collections.singletonList(measurement),
          Collections.singletonList(tsDataType), // NOTE UPDATE VALUE DATATYPE!
          parseValue("0", tsDataType)); // NOTE UPDATE VALUE DATATYPE!
      if (cntOther > 0) {
        for (int i = 1; i <= cntOther; i++) {
          session.insertRecord(
              otherDevice + i,
              1683616109697L, // ms
              // NOTE UPDATE TIME DATATYPE! [[update]]. DONT USE System.nanoTime()!
              Collections.singletonList(otherMeasurement),
              Collections.singletonList(tsDataType), // NOTE UPDATE VALUE DATATYPE!
              parseValue("0", tsDataType)); // NOTE UPDATE VALUE DATATYPE!
        }
      }
    }
    session.executeNonQueryStatement("flush");

    List<MeasurementSchema> schemaList = new ArrayList<>();
    schemaList.add(
        new MeasurementSchema(measurement, tsDataType, TSEncoding.valueOf(valueEncoding)));

    List<MeasurementSchema> otherSchemaList = new ArrayList<>();
    otherSchemaList.add(
        new MeasurementSchema(otherMeasurement, tsDataType, TSEncoding.valueOf(valueEncoding)));

    Tablet tablet = new Tablet(device, schemaList, iotdb_chunk_point_size);
    long[] timestamps = tablet.timestamps;
    Object[] values = tablet.values;

    Tablet otherTablet = new Tablet("tmp", otherSchemaList, iotdb_chunk_point_size);
    long[] otherTimestamps = otherTablet.timestamps;
    Object[] otherValues = otherTablet.values;

    File f = new File(filePath);
    String line = null;
    BufferedReader reader = new BufferedReader(new FileReader(f));
    reader.readLine(); // assume om3 table csv has table
    long globalTimestamp = -1;
    while (((line = reader.readLine()) != null) && (maxPointWritten < 0
        || globalTimestamp + 1 < maxPointWritten)) { // +1 because init -1
      String[] split = line.split(",");
      globalTimestamp++; // start from 0
      //  change to batch mode, iotdb_chunk_point_size
      int row = tablet.rowSize++;
      timestamps[row] = Long.parseLong(split[0]);
      double double_value = Double.parseDouble(split[valueIdx]); // get value from real data
      double[] double_sensor = (double[]) values[0];
      double_sensor[row] = double_value;
      if (tablet.rowSize == tablet.getMaxRowNumber()) { // chunk point size
        session.insertTablet(tablet, false);
        tablet.reset();
      }

      if (cntOther > 0) {
        row = otherTablet.rowSize++; // note ++
        otherTimestamps[row] = Long.parseLong(split[0]);
        double[] other_double_sensor = (double[]) otherValues[0];
        other_double_sensor[row] = double_value;
        if (otherTablet.rowSize == otherTablet.getMaxRowNumber()) { // chunk point size
          for (int i = 1; i <= cntOther; i++) {
            otherTablet.deviceId = otherDevice + i;
            session.insertTablet(otherTablet, false);
          }
          otherTablet.reset();
        }
      }

    }

    // flush the last Tablet
    if (tablet.rowSize != 0) {
      session.insertTablet(tablet, false);
      tablet.reset();
    }
    if (cntOther > 0) {
      if (otherTablet.rowSize != 0) {
        for (int i = 1; i <= cntOther; i++) {
          otherTablet.deviceId = otherDevice + i;
          session.insertTablet(otherTablet, false);
        }
        otherTablet.reset();
      }
    }
    session.executeNonQueryStatement("flush");
    session.close();
  }

  public static Object parseValue(String value, TSDataType tsDataType) throws IOException {
    if (tsDataType == TSDataType.DOUBLE) {
      return Double.parseDouble(value);
    } else {
      throw new IOException("data type wrong");
    }
  }
}
