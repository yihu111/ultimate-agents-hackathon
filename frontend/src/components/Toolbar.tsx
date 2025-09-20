import { ArrowLeft, Download, RotateCcw, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ToolbarProps {
  totalGenerated: number;
  selectedCount: number;
  selectedImages: string[];
  onRegenerateAll: () => void;
  onBack: () => void;
  onReset?: () => void;
}

export const Toolbar = ({
  totalGenerated,
  selectedCount,
  selectedImages,
  onRegenerateAll,
  onBack,
  onReset
}: ToolbarProps) => {
  const handleDownload = async () => {
    if (selectedImages.length === 0) return;

    try {
      // Download each selected image
      for (let i = 0; i < selectedImages.length; i++) {
        const imageUrl = selectedImages[i];
        const response = await fetch(imageUrl);
        const blob = await response.blob();
        
        // Create download link
        const downloadUrl = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = `ad-variation-${i + 1}.jpg`;
        
        // Trigger download
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Clean up object URL
        window.URL.revokeObjectURL(downloadUrl);
        
        // Small delay between downloads to prevent browser issues
        if (i < selectedImages.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  return (
    <div className="mb-6 rounded-lg border border-grid-border bg-grid-bg p-4">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        {/* Stats */}
        <div className="flex items-center space-x-6">
          <div className="text-sm">
            <span className="text-muted-foreground">Generated:</span>
            <span className="ml-2 font-semibold text-primary">{totalGenerated}</span>
          </div>
          <div className="text-sm">
            <span className="text-muted-foreground">Selected:</span>
            <span className="ml-2 font-semibold text-accent">{selectedCount}</span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center space-x-3">
          <Button
            variant="outline"
            onClick={onBack}
            className="bg-gradient-glass backdrop-blur-sm border-grid-border hover:bg-grid-hover"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>

          {onReset && (
            <Button
              variant="outline"
              onClick={onReset}
              className="bg-gradient-glass backdrop-blur-sm border-grid-border hover:bg-grid-hover"
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Reset
            </Button>
          )}

          <Button
            variant="outline"
            onClick={onRegenerateAll}
            className="bg-gradient-glass backdrop-blur-sm border-grid-border hover:bg-grid-hover"
          >
            <RotateCcw className="mr-2 h-4 w-4" />
            {selectedCount > 0 ? `Regenerate Selected (${selectedCount})` : 'Regenerate All'}
          </Button>

          <Button
            onClick={handleDownload}
            disabled={selectedCount === 0}
            className="bg-gradient-primary"
          >
            <Download className="mr-2 h-4 w-4" />
            Download ({selectedCount})
          </Button>
        </div>
      </div>
    </div>
  );
};