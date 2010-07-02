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
  ((analyze exp) env))

; The return value of analyze will be env -> succeed -> fail -> value
; succeed will be (lambda (value fail) ...)
;   then doesn't return! it just passes the value to succeed
; fail will be (lambda () ...)
; This means environment is not passed to succeed and fail
(define (ambeval exp env succeed fail)
  ((analyze exp) env succeed fail))


(define (analyze exp)
  (cond ((self-evaluating? exp)
         (analyze-self-evaluating exp))
        ((quoted? exp) (analyze-quoted exp))
        ((assignment? exp) (analyze-assignment exp))
        ((definition? exp) (analyze-definition exp))
        ((if? exp) (analyze-if exp))
        ((lambda? exp) (analyze-lambda exp))
        ((let? exp) (analyze (let->combination exp)))
        ((let*? exp) (analyze (let*->let exp)))
        ((begin? exp) (analyze-sequence (begin-actions exp)))
        ((cond? exp) (analyze (cond->if exp)))
        ((amb? exp) (analyze-amb exp))
        ((variable? exp) (analyze-variable exp))
        ((application? exp) (analyze-application exp))
        (else
         (error "Unknown expression type -- EVAL" exp))))

; Analyze returns function that can be applied to environment
; and produces value
(define (analyze-self-evaluating exp)
  (lambda (env succeed fail)
    (succeed exp fail)))

(define (analyze-quoted exp)
  (let ((qval (text-of-quotation exp)))
    (lambda (env succeed fail)
      (succeed qval fail))))

(define (analyze-variable exp)
    (lambda (env succeed fail) 
      (succeed (lookup-variable-value exp env) fail)))

(define (analyze-lambda exp)
  (let ((vars (lambda-parameters exp))
        (lbody (analyze-sequence (lambda-body exp))))
    (lambda (env succeed fail)
      (succeed (make-procedure vars lbody env) fail))))


(define (analyze-if exp)
  (let ((pproc (analyze (if-predicate exp)))
        (cproc (analyze (if-consequent exp)))
        (aproc (analyze (if-alternative exp))))
    (lambda (env succeed fail)
      (pproc
       env
       (lambda (val fail2)
         (if (eval-true? val)
             (cproc env succeed fail2)
             (aproc env succeed fail2)))
       fail))))

(define (analyze-sequence exps)
  (define (sequentially proc1 proc2)
    (lambda (env succeed fail)
      (proc1 env
             ; succeed continuation
             (lambda (val fail)
               (proc2 env succeed fail))
             ; failure continuation
             fail)))
  (define (loop first-proc rest-procs)
    (if (null? rest-procs)
        first-proc
        (loop (sequentially first-proc (car rest-procs))
              (cdr rest-procs))))
  (let ((procs (map analyze exps)))
    (if (null? procs)
        (error "Empty sequence -- ANALIZE"))
    (loop (car procs) (cdr procs))))

(define (analyze-assignment exp)
  (let ((var (assignment-variable exp))
        (vproc (analyze (assignment-value exp))))
    ; vproc must be applied to environment to produce value
    (lambda (env succeed fail)
      (vproc env
             (lambda (val fail2)
               (let ((old-val (lookup-variable-value var env)))
                 (set-variable-value! var val env)
                 (succeed 'ok
                          (lambda ()
                            (set-variable-value! var old-val env)
                            ; Don't pass environment to fail2?
                            (fail2)))))
             fail))))

(define (analyze-definition exp)
  (let ((var (definition-variable exp))
        (vproc (analyze (definition-value exp))))
    (lambda (env succeed fail)
      (vproc env
             (lambda (val fail2)
               (define-variable! var val env)
               (succeed 'ok fail2))
             fail))))


(define (analyze-application exp)
  (let ((fproc (analyze (operator exp)))
        (aprocs (map analyze (operands exp))))
    (lambda (env succeed fail)
      (fproc
       env
       (lambda (op fail2)
         (get-args
          aprocs
          env
          (lambda (operands fail2)
            (execute-application op operands succeed fail2))
          fail2))
       fail))))

; Continues passing the values to the succeed clause
(define (get-args aprocs env succeed fail)
  (if (null? aprocs)
      (succeed '() fail)
      (let ((aproc (car aprocs)))
        (aproc env
               (lambda (val fail2)
                 (get-args (cdr aprocs)
                           env
                           (lambda (vals fail3)
                             (succeed (cons val vals) fail3))
                           fail2))
               fail))))

; aprocs would be something like this?
(list
 (lambda (env succeed fail) (succeed 1 fail))
 (lambda (env succeed fail) (succeed 3 fail))
 (lambda (env succeed fail) (succeed 5 fail)))


(define (execute-application proc args succeed fail)
  (cond ((primitive-procedure? proc)
         (succeed (apply-primitive-procedure proc args) fail))
        ((compound-procedure? proc)
         ((procedure-body proc) ; this would be (lambda (env succeed fail) ...)
          (extend-environment (procedure-parameters proc)
                              args
                              (procedure-env proc))
          succeed
          fail))
        (else
         (error "Unknown procedure type -- EXECUTE-APPLICATION"
                proc))))

(define (amb? exp) (tagged-list? exp 'amb))

(define (amb-choices exp) (cdr exp))

; choices : (1 2 3)
; (success 1 (lambda () (success 2 (lambda () (success 3)))))
(define (analyze-amb exp)
  (let ((choices (amb-choices exp)))
    (lambda (env succeed fail)
      (define (iter choices)
        (if (null? choices)
            (fail)
            (succeed (car choices) ; repeat succeed
                     (lambda () (iter (cdr choices))))))
      (iter choices))))

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
(define (let-named-body exp) (cdddr exp))
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

(define (let-transform exp sym)
  (define (iter exp)
    (if (not (pair? exp))
        exp
        (if (and (symbol? (car exp)) (eq? sym (car exp)))
            (cons (car exp) (cons sym (iter (cdr exp))))
            (cons (iter (car exp)) (iter (cdr exp))))))
  (iter exp))

(let-transform '(hoge (tako (hoge 1) 2 3)) 'hoge)


(define (let->combination exp)
  (if (symbol? (cadr exp)) ; Named let '(let fun ((a b) (c d)) body)
      (let ((func-name (cadr exp))
            (vars (let-var-explicit (let-named-vars exp))))
        (let ((body (let-transform (let-named-body exp) func-name)))
        (let ((variables (cons func-name (var-names vars))))
          (let ((values (cons (make-lambda variables body)
                             (var-exps vars))))
            (cons (make-lambda variables body) ; cons means apply.
                  values)))))
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
        (list '= =)
        (list 'not not)
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
  (define (internal-loop try-again)
    (prompt-for-input input-prompt)
    (let ((input (read)))
      (if (eq? input 'try-again)
          (try-again)
          (begin
            (newline) (newline)
            (display ";;; Starting a new problem ")
            (ambeval input
                     the-global-environment
                     ;;
                     (lambda (val next-alternative) ; Still have the altenrative!
                       (announce-output output-prompt)
                       (user-print val)
                       (internal-loop next-alternative))
                     (lambda ()
                       (announce-output
                        ";;; There are no more values of ")
                       (user-print input)
                       (driver-loop)))))))
  (internal-loop
   (lambda ()
     (newline)
     (display ";;; There is no current problem")
     (driver-loop))))

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
(define (testEval input answer)
  (ambeval input the-global-environment
           (lambda (output next-alternative)
             (if (equal? output answer)
                 (display "ok! ")
                 (display "NG... ")))
           (lambda ()
             display 'failed)))

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
(testEval '(begin (define six 6) (define (mulseven x) (* 7 x)) (mulseven six)) 42)
(testEval '(begin (define (fib n)
                    (let fib-iter ((a 1) (b 0) (count n))
                      (if (= count 0)
                          b
                          (fib-iter (+ a b) a (- count 1)))))
                  (fib 5)) 5)
(testEval '(begin
             (define (require p) (if (not p) (amb)))
             (let ((a (amb 1 2 3))) (require (= a 2)) 2)) 2)
(testEval '(begin
             (define (require p) (if (not p) (amb)))
             (let ((a (amb 1 2 3)) (b (amb 7 8 9))) (require (= 12 (+ a b))) b)) 9)
; 0 1 1 2 3 5
