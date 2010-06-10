(define apply-in-underlying-scheme apply)

(define (primitive-procedure? proc)
  (tagged-list? proc 'primitive))

(define (primitive-implementation proc) (cadr proc))

(define (apply-primitive-procedure proc args)
  (apply-in-underlying-scheme
   (primitive-implementation proc) args))

;; apply function must precede eval
;; otherwise, the apply in oru eval is binded to native apply
(define (apply procedure arguments)
  (cond ((primitive-procedure? procedure)
         (begin
           (apply-primitive-procedure procedure arguments)))
        ((compound-procedure? procedure) ;; how about variable? -> eval.
         (begin
           (eval-sequence
            (procedure-body procedure)
            (extend-environment ; This returns new environment
             (procedure-parameters procedure)
             arguments
             (procedure-env procedure)))))
        (else
         (error "Unknown procedure type -- APPLY" procedure))))

(define (eval exp env)
  (cond ((self-evaluating? exp) exp)
        ((variable? exp) (lookup-variable-value exp env))
        ((quoted? exp) (text-of-quotation exp))
        ((assignment? exp) (eval-assignment exp env))
        ((definition? exp) (eval-definition exp env))
        ((if? exp) (eval-if exp env))
        ((cond? exp) (eval (cond->if exp) env))
        ((let*? exp) (eval (let*->let exp) env))
        ((let? exp) (eval (let->combination exp) env))
        ((lambda? exp)
         (make-procedure (lambda-parameters exp)
                         (lambda-body exp)
                         env))
        ((begin? exp)
         (eval-sequence (begin-actions exp) env))
        ((application? exp)
         (begin
           (apply (eval (operator exp) env)
                (list-of-values (operands exp) env))))
        (else
         (error "Unknown expression type -- EVAL" exp))))



(define (list-of-values exps env)
  ; Without the internal representation of exps,
  ; I should write the accessor functions
  (if (no-operands? exps)
      '()
      (cons (eval (first-operand exps) env)
            (list-of-values (rest-operands exps) env))))

(define (eval-if exp env)
  ; Metacircular representation of true might not the same as
  ; the underlying Scheme
  (if (eval-true? (eval (if-predicate exp) env))
      (eval (if-consequent exp) env)
      (eval (if-alternative exp) env)))

(define (eval-true? exp)
  (not (eval-null? exp)))

(define (eval-null? exp)
  (eq? exp '#f))

(define (eval-sequence exps env)
  (let ((ret (eval (first-exp exps) env)))
    (if (last-exp? exps)
        ret
        (eval-sequence (rest-exps exps) env))))

(define (eval-assignment exp env)
  (let ((variable (asignment-variable exp))
        (value (eval (assignment-exp exp) env)))
    (set-variable-value! variable value env))
  'ok)

(define (eval-definition exp env)
  (define-variable! (definition-variable exp)
    (eval (definition-value exp) env)
    env)
  'ok)


(define (self-evaluating? exp)
  (or (number? exp) (string? exp)
      (eq? exp '#f) (eq? exp '#t)
      (eq? exp '())))

(define (variable? exp)
  (symbol? exp))

; gosh> (quoted? '(quote a b))
; #t
(define (quoted? exp)
  (tagged-list? exp 'quote))

(define (text-of-quotation exp) (cadr exp))

; gosh> (tagged-list? '(a b) 'quote)
; #f
; gosh> (tagged-list? ''(a b) 'quote)
; #t
(define (tagged-list? exp tag)
  (and (pair? exp) (eq? (car exp) tag)))

; The expression is "tagged list"
; gosh> (assignment? '(set! a b))
; #t
(define (assignment? exp)
  (tagged-list? exp 'set!))

; (set! a b) => a
(define (assignment-variable exp)
  (cadr exp))

; (set! a b) => b
(define (assignment-exp exp)
  (caddr exp))

(define (definition? exp)
  (tagged-list? exp 'define))

; gosh> (symbol? (cadr '(define (a b) body)))
; #f
; gosh> (symbol? (cadr '(define a b body)))
; #t
(define (definition-variable exp)
  (if (symbol? (cadr exp))
      (cadr exp)
      (caadr exp)))

; gosh> (definition-value '(define a body))
; body
; gosh> (cdadr '(define (a b) body))
; (b)
(define (definition-value exp)
  (if (symbol? (cadr exp))
      (caddr exp)
      (make-lambda (cdadr exp) ; formal params
                   (cddr exp)))) ; body

; '(lambda (params-list) body)
(define (lambda? exp)
  (tagged-list? exp 'lambda))

(define (lambda-parameters exp)
  (cadr exp))

(define (lambda-body exp)
  (cddr exp))

(define (make-lambda parameters body)
  (cons 'lambda (cons parameters body)))

(define (if? exp) (tagged-list? exp 'if))


(define (if-predicate exp)
  (cadr exp))

;
; '(if p c a)
; '(if (p (c (a ()))))
(define (if-consequent exp)
  (caddr exp))

(define (if-alternative exp)
  (if (not (null? (cdddr exp)))
      (cadddr exp)
      'false))

(define (make-if predicate consequent alternative)
  (list 'if predicate consequent alternative))

(define (begin? exp) (tagged-list? exp 'begin))

(define (begin-actions exp) (cdr exp))

(define (last-exp? seq) (null? (cdr seq)))

(define (first-exp seq) (car seq))

(define (rest-exps seq) (cdr seq))

(define (sequence->exp seq)
  (cond ((null? seq) seq)
        ((last-exp? seq) (first-exp seq))
        (else (make-begin seq))))


(define (make-begin seq) (cons 'begin seq))

(define (application? exp)
  (pair? exp))

(define (operator exp) (car exp))

(define (operands exp) (cdr exp))

(define (no-operands? ops) (null? ops))

(define (first-operand ops) (car ops))

(define (rest-operands ops) (cdr ops))

(define (cond? exp) (tagged-list? exp 'cond))

(define (cond-clauses exp) (cdr exp))

(define (cond-else-clause? clause)
  (eq? (cond-predicate clause) 'else))

(define (cond-predicate clause)
  (car clause))

(define (cond-actions clause) (cdr clause))

(define (cond->if exp)
  (expand-clauses (cond-clauses exp)))

(define (expand-clauses clauses)
  (if (null? clauses)
      'false
      (let ((first (car clauses))
            (rest (cdr clauses)))
        (if (cond-else-lause? first)
            (if (null? rest)
                (sequence->exp (cond-actions first))
                (error "ELSE clause isn't last -- COND->IF"))
            (make-if (cond-predicate first)
                     (sequence->exp (cond-actions first))
                     (expand-clauses rest))))))

; '(let ((a b) (c d)) body)
(define (let? exp) (tagged-list? exp 'let))
(define (let-vars exp) (cadr exp))
(define (let-body exp) (cddr exp))
(define (let-named-vars exp) (caddr exp))
(define (let-named-vody exp) (cdddr exp))
(define (make-let vars body)
  (cons 'let (cons vars body)))

(define meta-nil 'metanil)

(define (let-var-explicit vars)
  (if (null? vars)
      ()
      (let ((h (car vars))
            (rest (cdr vars)))
        (if (symbol? h)
            (cons (list h meta-nil) (let-var-explicit rest))
            (cons h (let-var-explicit rest))))))

(define (let->combination exp)
  (if (symbol? (cadr exp)) ; Named let '(let fun ((a b) (c d)) body)
      (let ((func-name (cadr exp))
            (vars (let-var-explicit (let-named-vars exp)))
            (body (let-named-body exp)))
        (let ((variables (cons func-name (var-names vars))))
          (let ((values (cons (make-lambda variables body)
                             (var-exps vars))))
            (cons (make-lambda variables body)
                  values))))
      ; normal let '(let ((a b) (c d)) body)
      (let ((vars (let-var-explicit (let-vars exp)))
            (body (let-body exp)))
        (let ((variables (var-names vars))
              (values (var-exps vars)))
          (cons (make-lambda variables body)
            values)))))

; '((a 3) (b 5)) -> '(a b)
(define (var-names vars)
  (if (null? vars)
      ()
      (cons (caar vars) (var-names (cdr vars)))))

; ((a 3) (b 5)) -> (3 5)
(define (var-exps vars)
  (if (null? vars)
      ()
      (cons (cadar vars) (var-exps (cdr vars)))))

; '(let* ((a b) (c a))
(define (let*? exp) (tagged-list? exp 'let*))
(define (let*->let exp)
  (let ((vars (let-var-explicit (let-vars exp)))
        (body (let-body exp)))
    (define (iter vs)
      (if (last-exp? vs)
          (make-let (list (car vs)) body)
          (make-let (list (car vs)) (list (iter (cdr vs))))))
    (iter vars)))

;(let*->let '(let* ((a 5) (b a)) b))
;'(let ((a 5)) (let ((b a)) b))


(define (make-procedure parameters body env)
  (list 'procedure parameters body env))

(define (compound-procedure? p)
  (tagged-list? p 'procedure))

(define (procedure-parameters p)
  (cadr p))

(define (procedure-body p)
  (caddr p))

(define (procedure-env p)
  (cadddr p))

(define (lookup-variable-value var env)
  (define (env-loop env) ; for environment recursively
    (define (scan vars vals) ; for frame recursively
      (cond ((null? vars)
             (env-loop (enclosing-environment env)))
            ((eq? var (car vars))
             (car vals))
            (else (scan (cdr vars) (cdr vals)))))
    (if (eq? env the-empty-environment)
        (error "Unbound variable" var)
        (let ((frame (first-frame env)))
          (scan (frame-variables frame)
                (frame-values frame)))))
  (env-loop env))

(define (set-variable-value! var value env)
  (define (env-loop env)
    (define (scan vars vals)
      (cond ((null? vars)
             (env-loop (enclosing-environment env)))
            ((eq? var (car vars))
             (set-car vals value))
            (else (scan (cdr vars) (cdr (vals))))))
    (if (eq? env the-empty-environment)
        (error "Unbound variable" var)
        (let ((frame (first-frame env)))
          (scan (frame-variables frame)
                (frame-values frame)))))
  (env-loop env))


(define (extend-environment vars vals base-env)
  (if (= (length vars) (length vals))
      (cons (make-frame vars vals) base-env)
      (if (< (length vars) (length vals))
          (error "Too many arguments supplied -- EXTEND-ENVIRONMENT" vars vals)
          (error "Too few arguments supplied -- EXTEND-ENVIRONMENT" vars vals))))

; This doesn't make error when the var is not defined yet.
(define (define-variable! var value env)
  (let ((frame (first-frame env)))
    (define (scan vars vals)
      (cond ((null? vars)
             (add-binding-to-frame! var value frame))
            ((eq? var (car vars))
             (set-car! vals value))
            (else (scan (cdr vars) (cdr vals)))))
    (scan (frame-variables frame)
          (frame-values frame))))



; Does env have only one frame?
(define (enclosing-environment env) (cdr env))
(define (first-frame env) (car env))
(define the-empty-environment '())

; frame
; ((a,    b,    c)
;  (val1, val2, val3))
(define (make-frame variables values)
  (cons variables values))

(define (frame-variables frame) (car frame))
(define (frame-values frame) (cdr frame))

(define (add-binding-to-frame! var val frame)
  (set-car! frame (cons var (car frame)))
  (set-cdr! frame (cons val (cdr frame))))


(define primitive-procedures
  (list (list 'car car)
        (list 'cdr cdr)
        (list 'cons cons)
        (list 'null? null?)
        (list '+ +)
        (list '* *)
        (list '- -)
        (list '> >)
        (list '< <)
        (list '>= >=)
        (list '<= <=)
        ))

(define (primitive-procedure-names)
  (map car primitive-procedures))

(define (primitive-procedure-objects)
  (map (lambda (proc) (list 'primitive (cadr proc)))
       primitive-procedures))

(define (setup-environment)
  (let ((initial-env
         (extend-environment (primitive-procedure-names)
                             (primitive-procedure-objects)
                             the-empty-environment)))
    (define-variable! 'true #t initial-env)
    (define-variable! 'false #f initial-env)
    initial-env))

(define the-global-environment (setup-environment))


(define input-prompt ";;; M-Eval imput;")
(define output-prompt ";;; M-Eval value:")

(define (driver-loop)
  (prompt-for-input input-prompt)
  (let ((input (read)))
    (let ((output (eval input the-global-environment)))
      (announce-output output-prompt)
      (user-print output)))
  (driver-loop))

(define dl driver-loop)

(define (prompt-for-input string)
  (newline) (newline) (display string) (newline))

(define (announce-output string)
  (newline) (display string) (newline))

(define (user-print object)
  (if (compound-procedure? object)
      (display (list 'compound-procedure
                     (procedure-parameters object)
                     (procedure-body object)
                     '<procedure-env>))
      (display object)))

; tests

;; (let ((input '(+ 4 5))
;;       (answer 8))
;;   (let ((output (eval input the-global-environment)))
;;     (display answer) (newline)
;;     (if (number? output)
;;         (display "Thsi is number")
;;         (display "This is not number"))
;;     (if (equal? output answer)
;;         (display "ok")
;;         (display "ng"))))
(define (testEval input answer)
  (let ((output (eval input the-global-environment)))
    (if (equal? output answer)
        (display "ok! ")
        (display "NG... "))))
(testEval '(+ 4 5) 9)
(testEval '(let ((a 5) (b 3)) (+ a b)) 8)
(testEval '(cons 1 2) '(1 . 2))
(testEval '(car (cons 5 7)) 5)
(testEval '(cdr (cons 5 7)) 7)
(testEval '(if #t 19 23) 19)
(testEval '(if #f 19 23) 23)
(testEval '(if () 19 23) 19)
(testEval '(if (> 5 2) 20 29) 20)
(testEval '((lambda (a b) (+ a b)) 3 5) 8)
(testEval '(let ((k (lambda (a b) (let ((c 3)) (+ (- b a) c))))) (k 5 10)) 8)
(testEval '(let* ((a 5) (b (* a 3))) (+ b 1)) 16)

