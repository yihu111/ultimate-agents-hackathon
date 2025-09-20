import { ArrowLeft, Download, RotateCcw, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ToolbarProps {
  totalGenerated: number;
  selectedCount: number;
  onRegenerateAll: () => void;
  onBack: () => void;
  onReset?: () => void;
}

export const Toolbar = ({
  totalGenerated,
  selectedCount,
  onRegenerateAll,
  onBack,
  onReset
}: ToolbarProps) => {
  const handleDownload = () => {
    // TODO: Implement download functionality
    console.log('Download selected images');
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