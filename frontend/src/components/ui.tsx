import type {
  ButtonHTMLAttributes,
  InputHTMLAttributes,
  ReactNode,
  TextareaHTMLAttributes,
} from 'react'
import { ApiError } from '../api/client'
import { nl } from '../i18n/nl'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'
const variants: Record<Variant, string> = {
  primary: 'bg-brand text-accent-ink glow hover:brightness-110',
  secondary: 'bg-surface-2 text-ink hover:bg-line',
  ghost: 'text-ink-2 hover:bg-surface-2',
  danger: 'text-danger hover:bg-surface-2',
}

export function Button({
  variant = 'primary',
  className = '',
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      className={`inline-flex min-h-11 items-center justify-center gap-2 rounded-xl px-4 font-medium transition disabled:opacity-50 ${variants[variant]} ${className}`}
      {...props}
    />
  )
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <label className="block space-y-1.5">
      <span className="text-sm font-medium text-ink-2">{label}</span>
      {children}
      {hint && <span className="block text-xs text-ink-3">{hint}</span>}
    </label>
  )
}

const inputClass =
  'w-full rounded-xl border border-line bg-surface px-3 py-2.5 text-ink outline-none placeholder:text-ink-3 focus:border-accent focus:ring-2 focus:ring-accent/25'

// An extra className is added to the base style (it used to replace it).
export function Input({ className = '', ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={`${inputClass} ${className}`} {...props} />
}

export function TextArea({
  className = '',
  ...props
}: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className={`${inputClass} min-h-24 resize-y ${className}`} {...props} />
}

export function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`rounded-2xl border border-line bg-surface/90 p-4 shadow-lg shadow-black/20 transition hover:border-accent/30 ${className}`}
    >
      {children}
    </div>
  )
}

export function PageTitle({ children, action }: { children: ReactNode; action?: ReactNode }) {
  return (
    <div className="mb-4 flex items-center justify-between gap-3">
      <h1 className="text-brand text-3xl font-bold tracking-tight">{children}</h1>
      {action}
    </div>
  )
}

export function Spinner() {
  return (
    <div className="flex justify-center py-10 text-ink-3" role="status">
      {nl.common.loading}
    </div>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <p className="rounded-2xl border border-dashed border-line p-6 text-center text-ink-3">
      {children}
    </p>
  )
}

export function ErrorText({ error }: { error: unknown }) {
  if (!error) return null
  const message =
    error instanceof ApiError && typeof error.detail === 'string' ? error.detail : nl.common.error
  return (
    <p className="rounded-xl bg-warn-bg px-3 py-2 text-sm text-warn" role="alert">
      {message}
    </p>
  )
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  label,
}: {
  options: { value: T; label: string; hint?: string }[]
  value: T | null
  onChange: (v: T) => void
  label: string
}) {
  return (
    <div role="radiogroup" aria-label={label} className="flex flex-wrap gap-2">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          title={o.hint}
          className={`min-h-10 rounded-full border px-3.5 text-sm transition ${
            value === o.value
              ? 'border-transparent bg-brand text-accent-ink glow'
              : 'border-line bg-surface text-ink-2 hover:bg-surface-2'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}
