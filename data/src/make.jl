# Convert the original example datasets to compressed CSV/JSON files

# See data/README.md for the sources of the input data files
# To regenerate the output files:
# 1) Have all input files ready in the data folder
# 2) Instantiate the package environment for data/src
# 3) Run this script with the root folder of the repository being the working directory

using CSV, CodecZlib, JSON3, MAT

function bayes()
    data = CSV.File("data/data_bayes.csv")
    open(GzipCompressorStream, "data/bayes.csv.gz", "w") do stream
        CSV.write(stream, data)
    end
end

function sw()
    data = CSV.File("data/data_sw.csv")
    open(GzipCompressorStream, "data/sw.csv.gz", "w") do stream
        CSV.write(stream, data)
    end
end

function vlw()
    para = matread("data/modelparm_37sec.mat")
    tfp = matread("data/inddat_TFP_37sec.mat")
    out = Dict{Symbol,Any}()
    out[:δ] = para["moddel"]
    out[:ρA] = para["modrho"]
    out[:vash] = para["modvash"]
    out[:ksh] = para["modcapsh"]
    out[:iomat] = para["modiomat"]
    out[:csh] = para["modconssh"]
    out[:invmat] = para["modinvmat"]
    out[:εA] = tfp["ar1resid_GO"]

    open(GzipCompressorStream, "data/vlw.json.gz", "w") do stream
        JSON3.write(stream, out)
    end
end

function main()
    bayes()
    sw()
    vlw()
end

main()
