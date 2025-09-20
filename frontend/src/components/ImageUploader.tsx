import { useState, useCallback } from 'react';
import { Link as LinkIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';

interface ImageUploaderProps {
  onStart: (imageUrl: string, context?: string) => void;
  isLoading?: boolean;
}

export const ImageUploader = ({ onStart, isLoading }: ImageUploaderProps) => {
  const [imageUrl, setImageUrl] = useState('');
  const [context, setContext] = useState('');
  const { toast } = useToast();

  const handleStart = useCallback(() => {
    try {
      if (!imageUrl.trim()) {
        toast({ title: 'Missing image URL', description: 'Please enter a valid image URL.', variant: 'destructive' });
        return;
      }
      onStart(imageUrl.trim(), context);
    } catch (error) {
      console.error('Start failed:', error);
      toast({ title: 'Failed to start', description: 'Please check the image URL and try again.', variant: 'destructive' });
    }
  }, [imageUrl, context, onStart, toast]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-6">
      <div className="w-full max-w-2xl">
        <div className="mb-8 text-center">
          <h1 className="mb-4 text-5xl font-bold bg-gradient-primary bg-clip-text text-transparent">
            AdGrid
          </h1>
          <p className="text-xl text-muted-foreground">
            Paste an image URL and generate multiple ad variations
          </p>
        </div>

        {/* Context Input */}
        <div className="mb-6">
          <label htmlFor="context" className="block text-sm font-medium text-foreground mb-2">
            Product Context (Optional)
          </label>
          <textarea
            id="context"
            value={context}
            onChange={(e) => setContext(e.target.value)}
            placeholder="Describe your product, target audience, or any specific requirements for the ad variations..."
            className="w-full p-3 rounded-lg border border-grid-border bg-grid-bg text-foreground placeholder:text-muted-foreground resize-none focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-colors"
            rows={3}
            disabled={isLoading}
          />
        </div>

        <div className={cn(
          "relative rounded-lg border border-grid-border bg-grid-bg p-6 transition-all duration-300",
          isLoading && "animate-pulse-glow"
        )}>
          <div className="space-y-4">
            <label htmlFor="image-url" className="block text-sm font-medium text-foreground">Image URL</label>
            <div className="flex gap-2">
              <input
                id="image-url"
                type="url"
                placeholder="https://example.com/image.jpg"
                className="flex-1 p-3 rounded-lg border border-grid-border bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-colors"
                value={imageUrl}
                onChange={(e) => setImageUrl(e.target.value)}
                disabled={isLoading}
              />
              <Button onClick={handleStart} disabled={isLoading || !imageUrl.trim()} className="flex items-center gap-2">
                <LinkIcon className="h-4 w-4" />
                Generate
              </Button>
            </div>
          </div>
        </div>

        <div className="mt-6 text-center text-sm text-muted-foreground">
          Your image URL will be used directly to generate variations; no uploads
        </div>
      </div>
    </div>
  );
};