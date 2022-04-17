# Convert the original example datasets to compressed CSV files

# See data/README.md for the sources of the input data files
# To regenerate the output files:
# 1) Have all input files ready in the data folder
# 2) Instantiate the package environment for data/src
# 3) Run this script with the root folder of the repository being the working directory

using CSV, CodecZlib

function bayes()
    data = CSV.File("data/data_bayes.csv")
    open(GzipCompressorStream, "data/bayes.csv.gz", "w") do stream
        CSV.write(stream, data)
    end
end

function main()
    bayes()
end

main()
