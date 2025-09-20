import { useState } from 'react';
import { ImageUploader } from '@/components/ImageUploader';
import { AdGrid } from '@/components/AdGrid';

const Index = () => {
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [uploadResponse, setUploadResponse] = useState<any>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleImageSelect = (file: File, uploadResponse: any, context?: string) => {
    setIsProcessing(true);
    setUploadedImage(file);
    setUploadResponse({ ...uploadResponse, prompt: context });
    setIsProcessing(false);
  };

  const handleBack = () => {
    setUploadedImage(null);
    setUploadResponse(null);
    setIsProcessing(false);
  };

  if (uploadedImage && uploadResponse) {
    return (
      <AdGrid 
        uploadedImage={uploadedImage} 
        uploadResponse={uploadResponse} 
        onBack={handleBack}
        initialPrompt={uploadResponse.prompt}
      />
    );
  }

  return (
    <ImageUploader
      onImageSelect={handleImageSelect}
      isLoading={isProcessing}
    />
  );
};

export default Index;
