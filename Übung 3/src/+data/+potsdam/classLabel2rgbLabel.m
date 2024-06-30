function [rgbLabelImage] = classLabel2rgbLabel(labelImage)
%RGB2LABEL Summary of this function goes here
%   Detailed explanation goes here

% Label -> class name -> RGB:
%    1: Impervious surfaces (RGB: 255, 255, 255)
%    2: Building (RGB: 0, 0, 255)
%    3: Low vegetation (RGB: 0, 255, 255)
%    4: Tree (RGB: 0, 255, 0)
%    5: Car (RGB: 255, 255, 0)
%    6: Clutter/background (RGB: 255, 0, 0)

    imp_surf = labelImage == 1;
    building = labelImage == 2;
    low_veg  = labelImage == 3;
    tree     = labelImage == 4;
    car      = labelImage == 5;
    clutter  = labelImage == 6;
    
    r = imp_surf | car | clutter;
    g = imp_surf | low_veg | tree | car;
    b = imp_surf | building | low_veg;
    
    rgbLabelImage = cat(3,r,g,b) * 255;

end

