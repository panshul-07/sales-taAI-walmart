import { cn } from '../../lib/utils';

const variants = {
  default: 'ui-badge',
  subtle: 'ui-badge ui-badge-subtle',
};

export function Badge({ className, variant = 'default', ...props }) {
  return <span className={cn(variants[variant] || variants.default, className)} {...props} />;
}
