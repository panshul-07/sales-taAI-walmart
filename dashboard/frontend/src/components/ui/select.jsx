import { cn } from '../../lib/utils';

export function Select({ className, children, ...props }) {
  return (
    <div className="ui-select-wrap">
      <select className={cn('ui-select', className)} {...props}>
        {children}
      </select>
      <span className="ui-select-caret" aria-hidden>
        v
      </span>
    </div>
  );
}
