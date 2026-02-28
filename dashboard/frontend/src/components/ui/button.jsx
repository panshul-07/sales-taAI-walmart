import { cn } from '../../lib/utils';

const variants = {
  default: 'ui-button ui-button-default',
  outline: 'ui-button ui-button-outline',
  ghost: 'ui-button ui-button-ghost',
  secondary: 'ui-button ui-button-secondary',
};

export function Button({ className, variant = 'default', type = 'button', ...props }) {
  return <button type={type} className={cn(variants[variant] || variants.default, className)} {...props} />;
}
