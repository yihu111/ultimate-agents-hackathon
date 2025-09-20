import { useState, useCallback } from 'react';
import { Upload, ImageIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { apiService } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

interface ImageUploaderProps {
  onImageSelect: (file: File, uploadResponse: any) => void;
  isLoading?: boolean;
}

export const ImageUploader = ({ onImageSelect, isLoading }: ImageUploaderProps) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const { toast } = useToast();

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const imageFile = files.find(file => file.type.startsWith('image/'));
    
    if (imageFile) {
      await handleFileUpload(imageFile);
    }
  }, []);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      await handleFileUpload(file);
    }
  }, []);

  const handleFileUpload = async (file: File) => {
    try {
      const uploadResponse = await apiService.uploadImage(file);
      onImageSelect(file, uploadResponse);
    } catch (error) {
      console.error('Upload failed:', error);
      toast({
        title: "Upload Failed", 
        description: "Failed to upload image. Please try again.",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-6">
      <div className="w-full max-w-2xl">
        <div className="mb-8 text-center">
          <h1 className="mb-4 text-5xl font-bold bg-gradient-primary bg-clip-text text-transparent">
            AdGrid
          </h1>
          <p className="text-xl text-muted-foreground">
            Upload your product image and generate multiple ad variations
          </p>
        </div>

        <div
          className={cn(
            "relative rounded-lg border-2 border-dashed border-grid-border bg-grid-bg p-12 transition-all duration-300",
            isDragOver && "border-primary bg-grid-hover shadow-glow",
            isLoading && "animate-pulse-glow"
          )}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="flex flex-col items-center justify-center space-y-6">
            <div className="rounded-full bg-gradient-primary p-6">
              {isLoading ? (
                <div className="animate-spin">
                  <Upload className="h-12 w-12 text-primary-foreground" />
                </div>
              ) : (
                <ImageIcon className="h-12 w-12 text-primary-foreground" />
              )}
            </div>

            <div className="text-center">
              <h3 className="text-2xl font-semibold mb-2">
                {isLoading ? 'Processing your image...' : 'Drop your product image here'}
              </h3>
              <p className="text-muted-foreground mb-6">
                {isLoading ? 'Generating ad variations' : 'Supports PNG and JPG files up to 10MB'}
              </p>

              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                onChange={handleFileSelect}
                className="hidden"
                id="image-upload"
                disabled={isLoading}
              />
              
              <Button
                asChild
                variant="outline"
                size="lg"
                disabled={isLoading}
                className="bg-gradient-glass backdrop-blur-sm border-grid-border hover:bg-grid-hover"
              >
                <label htmlFor="image-upload" className="cursor-pointer">
                  <Upload className="mr-2 h-5 w-5" />
                  Choose File
                </label>
              </Button>
            </div>
          </div>
        </div>

        <div className="mt-6 text-center text-sm text-muted-foreground">
          Your image will be processed securely and variations will be generated instantly
        </div>
      </div>
    </div>
  );
};