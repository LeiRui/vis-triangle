# Experimental Guidance

## Preparations

### Install Packages

```shell
sudo apt-get update
sudo apt-get upgrade

apt install unzip
pip install simplification
pip install matplotlib
pip install pandas
pip install scipy
pip install opencv-python
pip install scikit-image
pip install opencv-python-headless
pip install scikit-learn
```

### Install Java

Java >= 1.8 is needed. Please make sure the JAVA_HOME environment path has been set. You can follow the steps below to install and configure Java.

```shell
sudo apt-get update
sudo apt-get upgrade

sudo apt install openjdk-8-jdk-headless

# configure
vim /etc/profile
# add the following two lines to the end of /etc/profile
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre
export PATH=$JAVA_HOME/bin:$PATH
# save and exit vim, and let the configuration take effect
source /etc/profile
```

Most experiments can run on Java 8, but the MinMaxCache experiment requires Java 11:

```bash
sudo apt install openjdk-11-jdk-headless

# configure
vim /etc/profile
# add the following two lines to the end of /etc/profile
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
# save and exit vim, and let the configuration take effect
source /etc/profile
```

### Download lts-exp

The structure of this repository is as follows:

-   `bash`: Folder of scripts for running experiments.
-   `iotdb-cli-0.12.4`: Folder of the IoTDB client.
-   `iotdb-server-0.12.4`: Folder of the IoTDB server.
-   `jarCode`: Folder of JAVA source codes for jars used in experiments.
-   `jars`: Folder of jars used in experiments to write data to IoTDB and query data from IoTDB.
-   `notebook`: Folder of the Python Jupyter Notebooks mainly for experiments on visual quality.
-   `python-exp`: Folder for the response time decomposition experiment involving remote connections.
-   `raspberryPi`: Folder for the battery life experiment.
-   `tools`: Folder of tools to assist automated experiment scripts.

### Download Datasets from Kaggle

First, create an empty folder named `datasets` within the downloaded `lts-exp` repository. This folder will hold the datasets used in the experiments. Then, download the datasets into the `datasets` folder by following the instructions below. 

The datasets are available in https://www.kaggle.com/datasets/anonymous1111111/pvdatasets. Here is the method of downloading data from kaggle on Ubuntu.

```shell
# First install kaggle.
pip install kaggle
pip show kaggle 

# Then set up kaggle API credentials.
mkdir ~/.kaggle # or /root/.kaggle
cd ~/.kaggle # or /root/.kaggle
vim kaggle.json # input your Kaggle API, in the format of {"username":"xx","key":"xx"}

# Finally you can download datasets.
# Assume that the downloaded path of lts-exp is /root/lts-exp.
cd /root/lts-exp/datasets
kaggle datasets download ANONYMOUS1111111/pvdatasets
unzip pvdatasets.zip # csv files will appear under /root/lts-exp/datasets
```

-   Qloss-small.csv: charge deficit of UPS battery, one month of data from the ground array. 
-   Qloss.csv: charge deficit of UPS battery, four months of data from the ground array. 
-   Pyra1-small.csv: millivolt output from domed-diffused silicon-cell pyranometer, one month of data from the ground array. 
-   Pyra1.csv: millivolt output from domed-diffused silicon-cell pyranometer, four months of data from the ground array. 
-   WindSpeed-small.csv: wind speed,  one month of data from the ground array. 
-   WindSpeed.csv: wind speed, four months of data from the ground array. 
-   RTD-small.csv: temperature of the module backsheet (south-center), one month of data from the ground array. 
-   RTD.csv: temperature of the module backsheet (south-center), four months of data from the ground array. 
-   RTD-more.csv: temperature of the module backsheet (center&northwest&south-center), four years of data from the ground array.
-   experiments.jar: used in the MinMaxCache experiment. Due to the large size of this jar file, it has not been directly placed in the Git repository, but is instead hosted here.



## Section 6.2 Accuracy Comparison

### Figure 10: Critical difference diagram on the UCR datasets

1.   Download the [UCR time series classification archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/). Suppose that the downloaded path is `/root/UCRArchive_2018`.
2.   Go to lts-exp/notebook, and run `python3 ucr-extract.py /root/UCRArchive_2018 /root/UCRsets-single `, where `/root/UCRArchive_2018` is the directory of the downloaded UCR datasets in the first step, and `/root/UCRsets-single` is the output directory of the concatenated long series. This step will take some time, please be patient.
3.   Go to lts-exp/jars, and run:
     -   `java -jar MySample_fsw_UCR-jar-with-dependencies.jar /root/UCRsets-single 800`: this will generate sampled time series by FSW with a target of 800 sampled points for all series in `/root/UCRsets-single`.
     -   `java -jar MySample_simpiece_UCR-jar-with-dependencies.jar /root/UCRsets-single 800`: this will generate sampled time series by Sim-Piece with a target of 800 sampled points for all series in `/root/UCRsets-single`.
4.   Go to lts-exp/notebook, and run `python3 ucr-test.py /root/UCRsets-single`. The experimental result is in `benchUCR.png`.



## Section 6.3 Visualization Comparison

### Figure 11: Visualizations of sampled time series

See [notebook/fig11-Visualizations of sampled time series.ipynb](notebook/fig11-Visualizations of sampled time series.ipynb).



## Section 6.4 Parameter Evaluation

### Figure 12: Effectiveness on the number of pixel columns w 

See [notebook/fig12-Effectiveness on the number of pixel columns w.ipynb](notebook/fig12-Effectiveness on the number of pixel columns w.ipynb).



### Figure 13: Effectiveness on the number of sampled points m 

See [notebook/fig13-Effectiveness on the number of sampled points m.ipynb](notebook/fig13-Effectiveness on the number of sampled points m.ipynb).



### Figure 14: Efficiency on the number of sampled points m 

1. Enter the `bash` folder within the downloaded `lts-exp` repository and then:

    -   Make all scripts executable by executing `chmod +x *.sh`.

    -   Update `prepare-all.sh` as follows:

        -   Update `TRI_VISUALIZATION_EXP` as the downloaded path of this `lts-exp` repository.

        -   Update `HOME_PATH` as an **empty** folder where you want the experiments to be executed.

    -   Run `prepare-all.sh` and then the folder at `HOME_PATH` will be ready for experiments. After this step is completed, "finish" will be printed on the screen.
2. Enter the folder at `HOME_PATH`, and run experiments using `nohup ./run-[datasetName]-efficiency-exp.sh 2>&1 &`, where `[datasetName]` is `Qloss`/`Pyra1`/`WindSpeed`/`RTD`. The running logs are saved in nohup.out, which can be checked by the command: `tail nohup.out`.
3. When the experiment script finishes running ("ALL FINISHED!" appears in nohup.out), the corresponding experimental results are in `HOME_PATH/res-[datasetName]-efficiency.csv`. 
4. In the result csv, counting from 1, the No.3 column is the query latency of `MinMax`, the No.8 column is the query latency of `M4`, the No.13 column is the query latency of `LTTB`, the No.18 column is the query latency of `MinMaxLTTB`, the No.23 column is the query latency of `ILTS`, the No.28 column is the query latency of `OM3`.
5. Plot results: see [notebook/fig14&15-Efficiency & Query latency breakdown.ipynb](notebook/fig14&15-Efficiency & Query latency breakdown.ipynb).



### Figure 15: Query latency breakdown of Figure 14 

See [notebook/fig14&15-Efficiency & Query latency breakdown.ipynb](notebook/fig14&15-Efficiency & Query latency breakdown.ipynb).




### Figure 16: Scalability on the number of input points n by varing the dataset size 

1. (If you have done this preparation step in the previous experiments, you can ignore it here and directly go to step 2.) Enter the `bash` folder within the downloaded `lts-exp` repository and then:

    -   Make all scripts executable by executing `chmod +x *.sh`.

    -   Update `prepare-all.sh` as follows:

        -   Update `TRI_VISUALIZATION_EXP` as the downloaded path of this `lts-exp` repository.

        -   Update `HOME_PATH` as an **empty** folder where you want the experiments to be executed.

    -   Run `prepare-all.sh` and then the folder at `HOME_PATH` will be ready for experiments.

2. Enter the folder at `HOME_PATH`, and run experiments using `nohup ./run-[datasetName]-scalability-byData-exp.sh 2>&1 &`, where `[datasetName]` is `Qloss`/`Pyra1`/`WindSpeed`/`RTD`. The running logs are saved in nohup.out, which can be checked by the command: `tail nohup.out`.

3. When the experiment script finishes running ("ALL FINISHED!" appears in nohup.out), the corresponding experimental results are in `HOME_PATH/res-[datasetName]-scalability-byData.csv`. 

4. In the result csv, counting from 1, the No.3 column is the query latency of `MinMax`, the No.8 column is the query latency of `M4`, the No.13 column is the query latency of `LTTB`, the No.18 column is the query latency of `MinMaxLTTB`, the No.23 column is the query latency of `ILTS`.



## Section 6.5 Application Evaluation

### Figure 17: Raspberry Pi battery level changes over time

Raspberry Pi is the database server, and a remote machine will receive the compressed query results.

1.   Install the Raspberry Pi battery monitoring module, e.g., https://www.waveshare.com/wiki/UPS_HAT.

2.   Fully charge the battery, as its depletion is not linear; start each experiment with the same initial charge level.

3.   Enter the `bash` folder within the downloaded `lts-exp` repository and then:

     -   Make all scripts executable by executing `chmod +x *.sh`. If you have done this step before, you can ignore it here.
     -   Update `prepare-energy-exp.sh` as follows:

         -   Update `TRI_VISUALIZATION_EXP` as the downloaded path of this `lts-exp` repository.
         -   Update `HOME_PATH` as an **empty** folder where you want the experiments to be executed.
     -   Run `prepare-energy-exp.sh` and then the folder at `HOME_PATH` will be ready for experiments.
     -   Enter the folder at `HOME_PATH`, and run experiments using `nohup ./run-energy-exp.sh 2>&1 &`. The running logs are saved in nohup.out, which can be checked by the command: `tail nohup.out`. When the experiment script finishes running ("ALL FINISHED!" appears in nohup.out), data preparations are complete.

4.   For the M4 sampling 10k experiment:

     -   Clear the cache: `echo 3 | sudo tee /proc/sys/vm/drop_caches`.

     -   Start the database server:

         -   Go to `HOME_PATH/iotdb-server-0.12.4/conf`, and update `enable_Tri=M4` in the iotdb-engine.properties.
         -   Go to `HOME_PATH/iotdb-server-0.12.4/sbin` and start the server: `./start-server.sh`.

     -   Delete the old items in `HOME_PATH` if they exist: `rm log*`, `rm -rf tmpDir*`.

     -   Go to `HOME_PATH`, and check the script `run.sh`:

         -   Update `approach` as `M4`.
         -   Update `m` as `10000`.

         -   Update `remote_ip` as the IP address of the receiving machine.

         -   Update `remote_user_name` as the login username of the receiving machine.


         -   Update `remote_passwd` as the login password of the receiving machine.
         -   Update `HOME_PATH` as the folder that you specified in the steps above.

     -   Run the experiment: `./run.sh`. At the same time, start the battery monitoring: go to `HOME_PATH`, delete the old `output.log` if it exists, and run `python3 INA219.py`. **The battery monitoring log will be written in `output.log`.**

     -   After monitoring for a sufficient duration (e.g., 4 hours), shut down the above running processes:

         -   Go to `HOME_PATH/iotdb-server-0.12.4/sbin` and stop the server: `./stop-server.sh`.
         -   Stop `run.sh`.
         -   Stop `INA219.py`.

5.   For the ILTS sampling 2k experiment:

     -   Clear the cache: `echo 3 | sudo tee /proc/sys/vm/drop_caches`.

     -   Start the database server:

         -   Go to `HOME_PATH/iotdb-server-0.12.4/conf`, and update `enable_Tri=ILTS` in the iotdb-engine.properties.
         -   Go to `HOME_PATH/iotdb-server-0.12.4/sbin` and start the server: `./start-server.sh`.

     -   Delete the old items if they exist: `rm log*`, `rm -rf tmpDir*`.

     -   Go to `HOME_PATH`, and check the script `run.sh`:

         -   Update `approach` as `ILTS`.
         -   Update `m` as `2000`.

         -   Update `remote_ip` as the IP address of the receiving machine.

         -   Update `remote_user_name` as the login username of the receiving machine.

         -   Update `remote_passwd` as the login password of the receiving machine.
         -   Update `HOME_PATH` as the folder that you specified in the steps above.

     -   Run the experiment: `./run.sh`. At the same time, start the battery monitoring: go to `HOME_PATH`, delete the old `output.log` if it exists, and run `python3 INA219.py`. **The battery monitoring log will be written in `output.log`.**

     -   After monitoring for a sufficient duration (e.g., 4 hours), shut down the above running processes:

         -   Go to `HOME_PATH/iotdb-server-0.12.4/sbin` and stop the server: `./stop-server.sh`.
         -   Stop `run.sh`.
         -   Stop `INA219.py`.



### Figure 18: ILTSCache v.s. MinMaxCache 

1.   This experiment requires Java 11. Please see the "Install Java" section in this README.
2.   Enter the `bash` folder within the downloaded `lts-exp` repository and then:
     -   Make all scripts executable by executing `chmod +x *.sh`.

     -   Update `prepare-cache-exp.sh` as follows:

         -   Update `TRI_VISUALIZATION_EXP` as the downloaded path of this `lts-exp` repository.

         -   Update `HOME_PATH` as an **empty** folder where you want the experiments to be executed.

     -   Run `prepare-cache-exp.sh` and then the folder at `HOME_PATH` will be ready for experiments.
3.   Enter the folder at `HOME_PATH`, and run experiments using `nohup ./run-cache-exp.sh 2>&1 &`. The running logs are saved in nohup.out, which can be checked by the command: `tail nohup.out`.
4.   When the experiment script finishes running ("ALL FINISHED!" appears in nohup.out), the corresponding experimental results are in `HOME_PATH/exp-cache-*`. 



## Section 6.6 Ablation Study (Figure 19-20) 

1. (If you have done this preparation step in the previous experiments, you can ignore it here and directly go to step 2.) Enter the `bash` folder within the downloaded `lts-exp` repository and then:

    -   Make all scripts executable by executing `chmod +x *.sh`.

    -   Update `prepare-all.sh` as follows:

        -   Update `TRI_VISUALIZATION_EXP` as the downloaded path of this `lts-exp` repository.

        -   Update `HOME_PATH` as an **empty** folder where you want the experiments to be executed.

    -   Run `prepare-all.sh` and then the folder at `HOME_PATH` will be ready for experiments.
2. Experiment with various m: Enter the folder at `HOME_PATH`, and run experiments using `nohup ./run-ablation-m-exp.sh 2>&1 &`. The running logs are saved in nohup.out, which can be checked by the command: `tail nohup.out`. When the experiment script finishes running ("ALL FINISHED!" appears in nohup.out), the corresponding experimental results are in `HOME_PATH/res-ablation-m.csv`.
3. Similarly, for experiment with various n: Enter the folder at `HOME_PATH`, and run experiments using `nohup ./run-ablation-n-exp.sh 2>&1 &`. The running logs are saved in nohup.out, which can be checked by the command: `tail nohup.out`. When the experiment script finishes running ("ALL FINISHED!" appears in nohup.out), the corresponding experimental results are in `HOME_PATH/res-ablation-n.csv`.
4. In the result csv, counting from 1, the No.3 to 7 columns are the results (i.e., query latency, I/O time cost, and count statistics) of `ILTS_a`, the No.8 to 12 columns are the results of `ILTS_b`, the No.13 to 17 columns are the results of `ILTS_c`, the No.18 to 22 columns are the results of `ILTS_d`.



## Section 5.1 & Figure 7: Visualizing time series from a remote database without/with in-database sampling queries 

This experiments involves communication between two nodes and is a bit more complicated than the previous sections in terms of installation preparation. Assume that the server and client nodes have the following IP addresses, usernames, and passwords.

|            | Database Server Node | Rendering Client Node |
| ---------- | -------------------- | --------------------- |
| IP address | A                    | B                     |
| Username   | server               | client                |
| Password   | x                    | y                     |

### (1) Environment Setup for Both Nodes

-   Download Java as instructed earlier.

-   Download `lts-exp` repository as instructed earlier.

-   Download sshpass:

    ```shell
    sudo apt-get install sshpass
    ```


-   **After downloading sshpass, run `sshpass -p 'x' ssh server@A "echo 'a'"` on the client node to verify if sshpass works. If sshpass works, you will see an "a" printed on the screen. Otherwise, try executing `ssh server@A "echo 'a'"` on the client node, and then reply "yes" to the prompt ("Are you sure you want to continue connecting (yes/no/[fingerprint])?") and enter the password 'x' manually. Then run again `sshpass -p 'x' ssh server@A "echo 'a'"` on the client node to verify if sshpass works.**

-   Download the Python packages to be used:

    ```shell
    sudo apt install python3-pip
    pip install matplotlib
    pip install thrift
    pip install pandas
    pip install pyarrow
    
    pip show matplotlib # this is to check where python packages are installed. 
    
    cd /root/lts-exp/python-exp
    # In the following, we assume that python packages are installed in "/usr/local/lib/python3.8/dist-packages"
    cp -r iotdb /usr/local/lib/python3.8/dist-packages/. # this step installs iotdb-python-connector
    ```



### (2) Populate the Database Server Node

Before doing experiments, follow the steps below to populate the database server with test data.

1. Go to the database server node.
2. Enter the `bash` folder within the downloaded `lts-exp` repository and then:

    -   Make all scripts executable by executing `chmod +x *.sh`. If you have done this step before, you can ignore it here.
    -   Update `prepare-motivation-exp.sh` as follows:
    
        -   Update `TRI_VISUALIZATION_EXP` as the downloaded path of this `lts-exp` repository.
        -   Update `HOME_PATH` as an **empty** folder where you want the experiments to be executed.
    -   Run `prepare-motivation-exp.sh` and then the folder at `HOME_PATH` will be ready for experiments.

3. Enter the folder at `HOME_PATH`, and run experiments using `nohup ./run-motivation-exp.sh 2>&1 &`.
    The running logs are saved in nohup.out, which can be checked by the command: `tail nohup.out`.

4. When the experiment script finishes running ("ALL FINISHED!" appears in nohup.out), preparations are complete.



### (3) Experiments on the Rendering Client Node

Go to the rendering client node. Enter the `python-exp` folder within the downloaded `lts-exp` repository and then:

1.   Make all scripts executable by executing `chmod +x *.sh`.

2.   Update `run-python-query-plot-exp.sh` as follows:

     -   Update `READ_METHOD` as `raw`/`lttb`/`ilts`.
         -   `raw`: corresponding to "Original" in Figure 18, i.e., using the raw data query at the database server.
         
         -   `lttb`: corresponding to "LTTB" in Figure 18, i.e., using the LTTB sampling query at the database server.

         -   `ilts`: corresponding to "ILTS" in Figure 18, i.e., using the ILTS sampling query with convex hull acceleration at the database server.
         
     -   Update `TRI_VISUALIZATION_EXP` as the downloaded path of the `lts-exp` repository on the client node.

     -   Update `remote_TRI_VISUALIZATION_EXP` as the downloaded path of the `lts-exp` repository on the server node.

     -   Update `remote_IOTDB_HOME_PATH` to the same path as the "HOME_PATH" set in the "(2) Populate the Database Server Node" section of this README.

     -   Update `remote_ip` as the IP address of the database server node.

     -   Update `remote_user_name` as the login username of the database server node.

     -   Update `remote_passwd` as the login password of the database server node.

3.   Run experiments using `nohup ./run-python-query-plot-exp.sh 2>&1 &`. The running logs are saved in nohup.out, which can be checked by the command: `tail nohup.out`. 

4.   When the experiment script finishes running ("ALL FINISHED!" appears in nohup.out), the corresponding experimental results are in `sumResult-[READ_METHOD].csv`, where `[READ_METHOD]` is `raw`/`lttb`/`ilts`. In the result csv, the last five columns are the `number of input data points`, `server computation time`, `communication time`, `client rendering time`, and `total response time`, respectively.





## Supplement: Scalability on the number of input points n by varing the query selectivity (Figure 23) 

1. (If you have done this preparation step in the previous experiments, you can ignore it here and directly go to step 2.) Enter the `bash` folder within the downloaded `lts-exp` repository and then:

    -   Make all scripts executable by executing `chmod +x *.sh`.

    -   Update `prepare-all.sh` as follows:

        -   Update `TRI_VISUALIZATION_EXP` as the downloaded path of this `lts-exp` repository.

        -   Update `HOME_PATH` as an **empty** folder where you want the experiments to be executed.

    -   Run `prepare-all.sh` and then the folder at `HOME_PATH` will be ready for experiments.

2. Enter the folder at `HOME_PATH`, and run experiments using `nohup ./run-[datasetName]-scalability-byQuery-exp.sh 2>&1 &`, where `[datasetName]` is `Qloss`/`Pyra1`/`WindSpeed`/`RTD`. The running logs are saved in nohup.out, which can be checked by the command: `tail nohup.out`.

3. When the experiment script finishes running ("ALL FINISHED!" appears in nohup.out), the corresponding experimental results are in `HOME_PATH/res-[datasetName]-scalability-byQuery.csv`. 

4. In the result csv, counting from 1, the No.3 column is the query latency of `MinMax`, the No.8 column is the query latency of `M4`, the No.13 column is the query latency of `LTTB`, the No.18 column is the query latency of `MinMaxLTTB`, the No.23 column is the query latency of `ILTS`.



## Supplement: Overhead Evaluation (Figure 24) 

1. (If you have done this preparation step in the previous experiments, you can ignore it here and directly go to step 2.) Enter the `bash` folder within the downloaded `lts-exp` repository and then:

    -   Make all scripts executable by executing `chmod +x *.sh`.

    -   Update `prepare-all.sh` as follows:

        -   Update `TRI_VISUALIZATION_EXP` as the downloaded path of this `lts-exp` repository.

        -   Update `HOME_PATH` as an **empty** folder where you want the experiments to be executed.

    -   Run `prepare-all.sh` and then the folder at `HOME_PATH` will be ready for experiments.
2. Enter the folder at `HOME_PATH`, and run experiments using `nohup ./run-overhead-exp.sh 2>&1 &`. The running logs are saved in nohup.out, which can be checked by the command: `tail nohup.out`.
3. When the experiment script finishes running ("ALL FINISHED!" appears in nohup.out), the corresponding experimental results are in running logs:
    -   "write latency of RTDMORE (without convex hull) for `x` is: `y` ns":  means that the write latency **without** convex hull precomputation for x% data is y ns.
    -   "write latency of RTDMORE (with convex hull) for `x` is: `y` ns":  means that the write latency **with** convex hull precomputation for x% data is y ns.
4. The space consumption can be checked using the command `du -s *` under the path `HOME_PATH/dataSpace_noConvexHull_[x]/RTDMORE_O_10_D_0_0/data/unsequence/root.RTDMORE/0` for **without** convex hull case, and `HOME_PATH/dataSpace_ConvexHull_[x]/RTDMORE_O_10_D_0_0/data/unsequence/root.RTDMORE/0` for **with** convex hull case, for writing x% data, x=20/40/60/80/100.



## Supplement: About ILTS parameter choices

### 1. The SSIM results of ILTS with random initialization are close to those with average initialization.

See [notebook/detail-ILTS with random initialization.ipynb](notebook/detail-ILTS with random initialization.ipynb).



### 2. ILTS converges within four iterations empirically (Figure 22).

See [notebook/detail-ILTS converges within four iterations.ipynb](notebook/detail-ILTS converges within four iterations.ipynb).



## More experimental details

### OM3 effectiveness & efficiency experiments

-   Use `tools/init_real_data.js` to write data into Postgres:

```shell
node init_real_data.js raw_data.Qloss100 Qloss.csv 10000000
node init_real_data.js raw_data.Pyra1100 Pyra1.csv 10000000
node init_real_data.js raw_data.WindSpeed100 WindSpeed.csv 10000000
node init_real_data.js raw_data.RTD100 RTD.csv 10000000
```

-   Follow the guidance from OM3 to perform the transformation, resulting in the transformed tables: `om3.qloss100_om3_16m`, `om3.pyra1100_om3_16m`, `om3.windspeed100_om3_16m`, and `om3.rtd100_om3_16m`.

-   Export the OM3 tables, each containing three columns: `i`, `minvd`, and `maxvd`. These will later reside in `lts-exp/datasets` and will be automatically written into IoTDB during the query efficiency experiment, using the first column `i` as the timestamp, with `minvd` and `maxvd` as the values of two time series.

```shell
COPY om3.qloss100_om3_16m TO 'qloss100_om3_16m.csv' WITH (FORMAT CSV, HEADER true);
COPY om3.pyra1100_om3_16m TO 'pyra1100_om3_16m.csv' WITH (FORMAT CSV, HEADER true);
COPY om3.windspeed100_om3_16m TO 'windspeed100_om3_16m.csv' WITH (FORMAT CSV, HEADER true);
COPY om3.rtd100_om3_16m TO 'rtd100_om3_16m.csv' WITH (FORMAT CSV, HEADER true);
```

-   Given a target number of `m` sampled points, we use a canvas width of `m/4` in the OM3 browser client, and collect logs during its visualization process. We then use `tools/parseOM3log.py` to extract queries in the log, summarizing all IDs from incremental queries, and record them in `tools/varyM`. Usually the number of IDs is larger than `m`.

    -   Later in the query efficiency experiment, we will use these IDs to generate corresponding IoTDB queries.

    -   Prior to the visual quality experiment, we generate the corresponding Postgres queries using `tools/genOM3sql.py`, and then export the queried OM3 table in the psql client (e.g., `\i 'om3-qloss.sql'`), saving them in `notebook/om3Tables`. Later in the visual quality experiment, we will reconstruct the time series from `notebook/om3Tables` by an inverse OM3 transform, and plot to compare SSIM.



### ILTSCache v.s. MinMaxCache experiment

-   The code we developed is available at the anonymized [link](https://anonymous.4open.science/r/minmaxcache-test-ADBF).
-   Running `mvn clean package` will generate `experiments.jar`, which we have included in the Kaggle dataset (see the "Download Datasets from Kaggle" section in this README).
-   We used the same experimental setup as described in [section 5.1 "Exploration Scenario" of the original MinMaxCache paper](https://www.vldb.org/pvldb/vol17/p2091-maroulis.pdf).
-   Since ILTSCache does not provide the error guarantees that MinMaxCache offers for guiding the adjustment of aggregation granularity, we used a fixed aggregation factor for ILTSCache.
