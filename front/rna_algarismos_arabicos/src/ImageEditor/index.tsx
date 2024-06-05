import React, { useState, useCallback } from 'react';
import { Box, Button, Typography, Slider } from '@mui/material';
import Dropzone from 'react-dropzone';
import Cropper from 'react-easy-crop';
import getCroppedImg from './cropImage.ts';
import { PredictionService } from '../api/predict.ts';
import { useQueryClient } from '@tanstack/react-query';


const ImageEditor: React.FC = () => {
  const [image, setImage] = useState<File | null>(null);
  const [croppedImage, setCroppedImage] = useState<string | null>(null);
  const [crop, setCrop] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const queryClient = useQueryClient();
  const { refetch, isLoading, isError, error, data } = PredictionService.useGetPrediction();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setImage(acceptedFiles[0]);
  }, []);

  const onCropComplete = useCallback(async (croppedArea: any, croppedAreaPixels: any) => {
    if (image) {
      const croppedImg = await getCroppedImg(image, croppedAreaPixels);
      setCroppedImage(croppedImg);
    }
  }, [image]);

  const handleUpload = async () => {
    if (croppedImage) {
      const blob = await fetch(croppedImage).then((res) => res.blob());
      const file = new File([blob], 'croppedImage.png', { type: 'image/png' });
      refetch(file as any);
    }
  };

  return (
    <Box>
      <Typography variant="h4">Image Editor</Typography>
      <Dropzone
        onDrop={onDrop}
        // accept="image/*"
      >
        {({ getRootProps, getInputProps }) => (
          <Box {...getRootProps()} border={1} padding={2} marginBottom={2}>
            <input {...getInputProps()} />
            <Typography>Drag 'n' drop an image here, or click to select one</Typography>
          </Box>
        )}
      </Dropzone>
      {image && (
        <Box position="relative" width="100%" height={400} marginBottom={2}>
          <Cropper
            image={URL.createObjectURL(image)}
            crop={crop}
            zoom={zoom}
            aspect={4 / 3}
            onCropChange={setCrop}
            onCropComplete={onCropComplete}
            onZoomChange={setZoom}
          />
          <Slider value={zoom} min={1} max={3} step={0.1} onChange={(e, zoom) => setZoom(zoom as number)} />
        </Box>
      )}
      {croppedImage && (
        <Box marginBottom={2}>
          <img src={croppedImage} alt="Cropped" style={{ maxWidth: '100%' }} />
        </Box>
      )}
      <Button variant="contained" color="primary" onClick={handleUpload} disabled={!croppedImage || isLoading}>
        Upload & Predict
      </Button>
      {isLoading && <Typography>Loading...</Typography>}
      {error && <Typography>Error: {error.message}</Typography>}
      {data && <Typography>Prediction: {data.predicted_label}</Typography>}
    </Box>
  );
};

export default ImageEditor;