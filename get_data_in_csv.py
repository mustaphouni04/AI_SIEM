import pandas as pd
from pandas import DataFrame

import pandas as pd
import re
import os

def get_data(logs_directory):
    # define a function to parse each log entry
    def parse_log_entry(log_entry):
        # define a regular expression pattern to extract information from the log entry
        pattern = r'(?P<server_name>[\w.-]+) (?P<remote_host>\S+) (?P<remote_logname>\S+) (?P<remote_user>\S+) \[(?P<timestamp>[^\]]+)\] "(?P<request_method>[A-Z]+) (?P<requested_url>\S+) HTTP/\d\.\d" (?P<status_code>\d+) (?P<bytes_sent>\d+) "(?P<referer>[^"]+)" "(?P<user_agent>[^"]+)"'
        
        # match the pattern against the log entry
        match = re.match(pattern, log_entry)
        
        if match:
            return match.groupdict()
        else:
            return None
    
    # list all files in the directory
    log_files = [f for f in os.listdir(logs_directory) if os.path.isfile(os.path.join(logs_directory, f))]
    
    # list to store parsed log entries
    all_log_entries = []
    
    # process each log file
    for log_file in log_files:
        log_file_path = os.path.join(logs_directory, log_file)
        
        # read log file line by line and parse each entry
        with open(log_file_path, 'r') as file:
            for line in file:
                log_entry = parse_log_entry(line.strip())
                if log_entry:
                    all_log_entries.append(log_entry)
    
    # convert list of dictionaries to DataFrame
    df = pd.DataFrame(all_log_entries)

    csv_file_path = "../combined_logs.csv"
    df.to_csv(csv_file_path, index=False)

    print(f"CSV file saved successfully: {csv_file_path}")

    return df