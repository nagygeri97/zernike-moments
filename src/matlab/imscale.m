function scaleRecovered = imscale(originalFile, distortedFile)
% Source: https://www.mathworks.com/help/vision/examples/find-image-rotation-and-scale-using-automated-feature-matching.html

path = "../../images/templates/small/";
outPath = "../../images/templates/scaled/";
originalPath = path + originalFile;
distortedPath = path + distortedFile;

originalColor = imread(originalPath);
distortedColor = imread(distortedPath);
original = rgb2gray(originalColor);
distorted = rgb2gray(distortedColor);

ptsOriginal  = detectSURFFeatures(original);
ptsDistorted = detectSURFFeatures(distorted);

[featuresOriginal,  validPtsOriginal]  = extractFeatures(original,  ptsOriginal);
[featuresDistorted, validPtsDistorted] = extractFeatures(distorted, ptsDistorted);

indexPairs = matchFeatures(featuresOriginal, featuresDistorted);

matchedOriginal  = validPtsOriginal(indexPairs(:,1));
matchedDistorted = validPtsDistorted(indexPairs(:,2));

% figure;
% showMatchedFeatures(original,distorted,matchedOriginal,matchedDistorted);
% title('Putatively matched points (including outliers)');

[tform, inlierDistorted, inlierOriginal] = estimateGeometricTransform(...
    matchedDistorted, matchedOriginal, 'similarity');

% figure;
% showMatchedFeatures(original,distorted,inlierOriginal,inlierDistorted);
% title('Matching points (inliers only)');
% legend('ptsOriginal','ptsDistorted');

Tinv  = tform.invert.T;

ss = Tinv(2,1);
sc = Tinv(1,1);
scaleRecovered = sqrt(ss*ss + sc*sc);
thetaRecovered = atan2(ss,sc)*180/pi;

outputView = imref2d(size(original));
recovered  = imwarp(distortedColor,tform,'OutputView',outputView);

% figure, imshowpair(originalColor,recovered,'montage');

% imwrite(recovered, outPath + distortedFile);
end





