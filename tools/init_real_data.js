const gaussian = require("gaussian");
const fs = require("fs");
const { Pool } = require('pg');
const csv = require('csv-parser');

const dbConfig = JSON.parse(fs.readFileSync("./dbconfig.json").toString());
if (!dbConfig['username'] || !dbConfig['hostname'] || !dbConfig['password'] || !dbConfig['db']) {
    throw new Error("db config error");
}

const pool = new Pool({
    user: dbConfig['username'],
    host: dbConfig['hostname'],
    database: dbConfig['db'],
    password: dbConfig['password'],
    port: dbConfig['port']
});

//const MAX_TOTAL_POINTS = Math.pow(2, 24);

async function createTable(fullTableName) {
    const [schemaName, tableName] = fullTableName.split('.');

    if (!schemaName || !tableName) {
        throw new Error(`Invalid table name format: ${fullTableName}`);
    }

    const checkTableExistsQuery = `
        SELECT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_tables
            WHERE schemaname = $1
            AND tablename = $2
        );
    `;

    try {
        const res = await pool.query(checkTableExistsQuery, [schemaName, tableName]);
        const tableExists = res.rows[0].exists;

        if (tableExists) {
            throw new Error(`Table ${fullTableName} already exists. Exiting.`);
        }

        const createTableQuery = `
            CREATE TABLE ${schemaName}.${tableName} (
                t INTEGER,
                v DOUBLE PRECISION
            );
        `;
        console.log(createTableQuery);
        await pool.query(createTableQuery);
        console.log(`Table ${fullTableName} created successfully.`);
    } catch (error) {
        console.error(`Error in table creation: ${error.message}.  Exiting...`);
        throw error;
    }
}


async function readCSVAndInsertToDB(csvFilePath, tableName, n1) {
    let data = [];
    const insertBatchSize = 10000; // Batch size for bulk insertions
    const insertPromises = [];
    let totalPoints = 0;
    let tCounter = 0; // Start t from 0

    const n2 = Math.pow(2, Math.ceil(Math.log2(n1)));

    fs.createReadStream(csvFilePath)
        .pipe(csv({ headers: false }))
        .on('data', (row) => {
            if (totalPoints >= n2) {
                // Stop processing if we've already reached the limit
                return;
            }
            const vValue = parseFloat(row['1']); // Ensure value is a float

            if (!isNaN(vValue)) {
                data.push({ t: tCounter, v: vValue });
                tCounter++; // Increment t for each row
                totalPoints++;
            }

            // When batch size is reached, insert batch
            if (data.length === insertBatchSize) {
                insertPromises.push(insertToDB([...data], tableName)); // Clone data to avoid mutation
                data = [];
            }
        })
        .on('end', async () => {
            // Insert remaining data if any
            if (data.length > 0) {
                insertPromises.push(insertToDB([...data], tableName));
            }

            // If totalPoints is less than n2, add additional points with v = null
            while (totalPoints < n2) {
                let extraData = [];
                while (extraData.length < insertBatchSize && totalPoints < n2) {
                    extraData.push({ t: tCounter, v: null });
                    tCounter++;
                    totalPoints++;
                }
                insertPromises.push(insertToDB([...extraData], tableName));
            }

            await Promise.all(insertPromises); // Wait for all insertions to finish
            await pool.end(); // Close the connection pool
            console.log('CSV file fully processed and data inserted in order.');
        });
}

async function insertToDB(data, tableName) {
    if (data.length === 0) return;

    // Generate the bulk insert query
    const values = data.map((item, i) => `($${i * 2 + 1}, $${i * 2 + 2})`).join(',');
    const queryText = `INSERT INTO ${tableName} (t, v) VALUES ${values}`;

    // Flatten the data into query parameters
    const queryParams = data.flatMap(d => [d.t, d.v]);

    try {
        await pool.query(queryText, queryParams);
        console.log(`Inserted ${data.length} rows.`);
    } catch (error) {
        console.error('Error inserting batch:', error);
        throw error;
    }
}

async function main() {
    // node xxx.js raw_data.Qloss D://desktop//NISTPV//Qloss.csv 10000000

    const tableName = process.argv[2];
    const csvPath = process.argv[3];
    const n1 = parseInt(process.argv[4], 10);

    try {
        await createTable(tableName);
        await readCSVAndInsertToDB(csvPath, tableName, n1);
    } catch (error) {
        console.error("Error in processing:", error);
    }
}

main();
