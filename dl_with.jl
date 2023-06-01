# Chapter 12 Julia Data Analysis Bogumil Kaminski
import Downloads
using SHA

git_zip = "git_web_ml.zip"

if !isfile(git_zip)
    Downloads.download("https://snap.stanford.edu/data/" * "git_web_ml.zip", git_zip)
end

isfile(git_zip) || error("Could not download git_web_ml.zip")

open(sha256, git_zip)

##################### opening zip file #####################
import ZipFile

git_archive = ZipFile.Reader(git_zip)

git_archive.files
# 1 directory with 5 files
# each stored file has several properties. Check the name property of the second file.

git_archive.files[2].name

# Create a helper function that creates a df from a csv file in an archive.

function ingest_to_df(archive::ZipFile.Reader, filename::AbstractString)
    idx = only(findall(x -> x.name == filename, archive.files))
    return CSV.read(read(archive.files[idx]), DataFrame)
end
# findall(x -> x.name == filename, archive.files) call finds all files whose name matches
# the filename variable and returns them as a vector.
# findall function takes two args. First is a function specifying a condition you want to check (whether name of the file matches `filename`)
# Second argumnet is a collection; from which, you want to find elements for which the function passed as the first argument returns true.
# findall returns a vector of indices to the collection for which the checked condition is satisfied.
# example of two findall calls:

findall(x -> x.name == "git_web_ml/musae_git_edges.csv", git_archive.files)