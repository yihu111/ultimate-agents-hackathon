import { useState } from 'react';
import { ImageUploader } from '@/components/ImageUploader';
import { AdGrid } from '@/components/AdGrid';

const Index = () => {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [initialPrompt, setInitialPrompt] = useState<string | undefined>(undefined);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleStart = (url: string, context?: string) => {
    setIsProcessing(true);
    setImageUrl(url);
    setInitialPrompt(context);
    setIsProcessing(false);
  };

  const handleBack = () => {
    setImageUrl(null);
    setInitialPrompt(undefined);
    setIsProcessing(false);
  };

  if (imageUrl) {
    return (
      <AdGrid 
        imageUrl={imageUrl}
        onBack={handleBack}
        initialPrompt={initialPrompt}
      />
    );
  }

  return (
    <ImageUploader onStart={handleStart} isLoading={isProcessing} />
  );
};

export default Index;
