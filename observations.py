'''
    1. feature detection struggles due to size invariance; namely, because it considers ratios of targeted features, it will search for feature patterns both large and small
        -- This is evident when using images with jagged rock cliffs; the cliffs create a number of different patterns that are detected as feature constellations
    2. Because this implementation uses a binary response, multiple faces in an frame negatively impact the general accuracy; if some faces have a mask and others do not, this implementation will always respond that the faces in the frame are unmasked
    3. This implementation does not distinguish between facial covering; in other words, anything covering the face (mask, towel, hand, etc ) will receive a "wearing mask" response
    4. This particular implementation seems to be heavily reliant on even shading across the face

    further work
    1. pass a specified region of interest into the feature constellation detection model
    2. 

'''
