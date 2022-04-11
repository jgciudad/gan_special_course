import SimpleITK as sitk
import json

# Tips:
# https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Utilities/intro_animation.py
# https://stackoverflow.com/questions/30237024/operate-on-slices-of-3d-image-in-simpleitk-and-create-new-3d-image
def read_aorta_landmarks():
    full_name = 'C:/data/VideoMaterial/ABD_LYMPH_001/F.mrk.json'
    f = open(full_name)

    # https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html
    # returns JSON object as a dictionary
    data = json.load(f)
    t = data['markups'][0]['controlPoints']

    first = True
    xmin = xmax = ymax = ymin = zmin = zmax = 0
    for lm in t:
        pos = lm['position']
        x = pos[0]
        y = pos[1]
        z = pos[2]
        if first:
            first = False
            xmin = xmax = x
            ymin = ymax = y
            zmin = zmax = z
        else:
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
            zmin = min(zmin, z)
            zmax = max(zmax, z)

    f.close()
    bounds = [xmin, xmax, ymin, ymax, zmin, zmax]
    return bounds


def crop_aorta_roi(bounds):
    full_name = 'C:/data/VideoMaterial/ABD_LYMPH_001/abdominal_lymph_nodes.nrrd'
    resampled_name = 'C:/data/test/abdominal_lymph_nodes_resampled.nrrd'
    image = sitk.ReadImage(full_name)
    # print(image.GetOrigin())
    # print(image.GetSize())
    # print(image.GetSpacing())
    # print(image.GetSize())
    # print(image.GetWidth())
    # print(image.GetHeight())
    # print(image.GetDepth())
    # print(image.GetPixelID())
    # print(image.GetNumberOfComponentsPerPixel())

    # Create the sampled image with same direction
    direction = image.GetDirection()

    # Desired voxel spacing for new image
    new_spacing = [0.25, 0.25, 0.25]

    # adjust bounds
    # Add some millimeters on each side
    padding = 15

    # in slice size (max of x length and y length plus padding in both sides)
    max_l = max(bounds[1]-bounds[0], bounds[3]-bounds[2]) + 2 * padding
    nvox_xy = int(max_l / new_spacing[0] + 1)
    new_l_xy = nvox_xy * new_spacing[0]
    nvox_z = int((bounds[5] - bounds[4] + 2 * padding) / new_spacing[2])
    print('Size of new volume: ', nvox_xy, nvox_xy, nvox_z, ' voxels')

    # Compute new origin from center of old bounds
    new_origin_x = (bounds[1] + bounds[0]) / 2 - new_l_xy / 2
    new_origin_y = (bounds[3] + bounds[2]) / 2 - new_l_xy / 2
    new_origin_z = (bounds[5] + bounds[4]) / 2 - nvox_z * new_spacing[2] / 2

    # Size in number of voxels per side
    # new_size = [100, 100, 100]
    new_size = [nvox_xy, nvox_xy, nvox_z]
    new_image = sitk.Image(new_size, image.GetPixelIDValue())
    new_image.SetOrigin([new_origin_x, new_origin_y, new_origin_z])
    new_image.SetSpacing(new_spacing)
    new_image.SetDirection(direction)

    # Make translation with no offset, since sitk.Resample needs this arg.
    translation = sitk.TranslationTransform(3)
    translation.SetOffset((0, 0, 0))

    interpolator = sitk.sitkLinear
    # Create final reasampled image
    resampled_image = sitk.Resample(image, new_image, translation, interpolator)

    sitk.WriteImage(resampled_image, resampled_name)


if __name__ == '__main__':
    bds = read_aorta_landmarks()
    crop_aorta_roi(bds)

