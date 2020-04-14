path = "../../images/templates/small/";
files = dir(path);
files = files';
files = files(3:end);
for file=files
    name = file.name;
    original = name(1:(end-5)) + "1.jpg";
    imscale(original, name);
end